import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Needs input shape [SeqLen, Batch, EmbedDim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class FeedForwardExpert(nn.Module):
    """A simple Feed-Forward Network expert."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VanillaRouter(nn.Module):
    """
    Simple vanilla router that maps token embeddings to expert logits using a linear layer.
    """
    def __init__(self, embedding_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(embedding_dim, num_experts)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: [B, M, T, E] or [B*M*T, E]
        Returns:
            router_logits: [B*M*T, num_experts]
        """
        # Flatten to [B*M*T, E] if needed
        if token_embeddings.dim() == 4:
            B, M, T, E = token_embeddings.shape
            token_embeddings = token_embeddings.view(B * M * T, E)
        
        return self.router(token_embeddings)


class BaselineMoELayer(nn.Module):
    """
    Baseline Mixture-of-Experts layer with vanilla router and feedforward experts only.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int,
                 k: int,
                 expert_hidden_dim: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        assert num_experts > 0 and 0 < k <= num_experts
        assert input_dim == output_dim, "MoE Layer requires input_dim == output_dim for residual connections"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k

        # All experts are feedforward experts
        self.experts = nn.ModuleList([
            FeedForwardExpert(input_dim, expert_hidden_dim, output_dim, dropout_rate)
            for _ in range(num_experts)
        ])

        # Simple vanilla router
        self.router = VanillaRouter(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, S, E_in = x.shape
        assert E_in == self.input_dim
        device = x.device
        num_tokens = B * S

        # Get router logits
        router_logits = self.router(x.view(num_tokens, E_in))  # [B*S, num_experts]
        gating_probs = F.softmax(router_logits, dim=-1)  # For load balancing loss

        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(router_logits, self.k, dim=-1)
        top_k_router_probs = F.softmax(top_k_weights, dim=-1)  # Actual routing weights

        tokens_flat = x.view(num_tokens, E_in)
        final_output_flat = torch.zeros(num_tokens, self.output_dim, device=device)
        
        # Process through experts
        flat_expert_indices = top_k_indices.view(-1)
        flat_router_probs = top_k_router_probs.view(-1)
        flat_batch_indices = torch.arange(num_tokens, device=device).repeat_interleave(self.k)

        for expert_idx in range(self.num_experts):
            mask = (flat_expert_indices == expert_idx)
            if mask.any():
                original_token_indices = flat_batch_indices[mask]
                current_routing_probs = flat_router_probs[mask].unsqueeze(-1)
                expert_input = tokens_flat[original_token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_expert_output = expert_output * current_routing_probs
                final_output_flat.index_add_(0, original_token_indices, weighted_expert_output)

        final_output = final_output_flat.view(B, S, self.output_dim)
        aux_outputs = {
            "gating_probs": gating_probs.view(B, S, self.num_experts),
            "expert_indices": top_k_indices.view(B, S, self.k),
        }
        return final_output, aux_outputs


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class BaselineMoEBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, moe_layer: BaselineMoELayer, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_layer = moe_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_moe = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        moe_output, aux_moe_outputs = self.moe_layer(src)
        src = src + self.dropout_moe(moe_output)
        src = self.norm2(src)
        return src, aux_moe_outputs


class BaselineMoEModel(nn.Module):
    def __init__(self,
                 input_dim: int, d_model: int, nhead: int, d_ff: int,
                 num_encoder_layers: int, num_moe_layers: int,
                 moe_config: Dict, num_modalities: int, num_classes: int,
                 dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        assert d_model == moe_config['input_dim'] and d_model == moe_config['output_dim']

        self.num_modalities = num_modalities
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList()

        # Determine which layers will be MoE layers
        moe_indices = set()
        if num_moe_layers > 0 and num_encoder_layers > 0:
            step = max(1, num_encoder_layers // num_moe_layers)
            indices_to_add = sorted([min(i * step, num_encoder_layers - 1) for i in range(num_moe_layers)])
            # Handle potential duplicates by shifting subsequent indices
            final_indices = []
            current_idx = -1
            for idx in indices_to_add:
                chosen_idx = max(idx, current_idx + 1)
                if chosen_idx < num_encoder_layers:
                    final_indices.append(chosen_idx)
                    current_idx = chosen_idx
            moe_indices = set(final_indices)
            # Add more if needed due to boundary conditions / collisions
            while len(moe_indices) < num_moe_layers:
                available = set(range(num_encoder_layers)) - moe_indices
                if not available: break
                moe_indices.add(random.choice(list(available)))

        print(f"Baseline MoE layers will be at indices: {sorted(list(moe_indices))}")

        for i in range(num_encoder_layers):
            if i in moe_indices:
                moe_layer_instance = BaselineMoELayer(**moe_config)
                self.layers.append(BaselineMoEBlock(d_model, nhead, moe_layer_instance, dropout))
            else:
                self.layers.append(TransformerBlock(d_model, nhead, d_ff, dropout))

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, token_embeddings: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        B, M, T, E_in = token_embeddings.shape
        S = M * T
        x = self.input_proj(token_embeddings) * math.sqrt(self.d_model)
        x = x.view(B, S, self.d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        all_aux_moe_outputs = []
        for layer in self.layers:
            if isinstance(layer, BaselineMoEBlock):
                x, aux_outputs = layer(x)
                all_aux_moe_outputs.append(aux_outputs)
            elif isinstance(layer, TransformerBlock):
                x = layer(x)
            else:
                raise TypeError("Unsupported layer type")

        x = self.final_norm(x)
        aggregated_output = x.mean(dim=1)  # Mean pooling over sequence S
        final_logits = self.output_proj(aggregated_output)
        return final_logits, all_aux_moe_outputs


def calculate_load_balancing_loss(gating_probs: torch.Tensor, expert_indices: torch.Tensor, k: int, lambda_load: float) -> torch.Tensor:
    """Calculates the load balancing loss (Switch Transformer version)."""
    if gating_probs.numel() == 0: 
        return torch.tensor(0.0, device=gating_probs.device)
    
    B, S, N_exp = gating_probs.shape
    num_tokens = B * S
    probs_flat = gating_probs.view(num_tokens, N_exp)  # P(expert|token)

    # f_i = fraction of tokens dispatched to expert i
    # Calculate how many times each expert was chosen in top-k
    expert_counts = torch.zeros(N_exp, device=gating_probs.device)
    expert_indices_flat = expert_indices.view(num_tokens, k)
    for i in range(k):
        counts = torch.bincount(expert_indices_flat[:, i], minlength=N_exp)
        expert_counts += counts
    f_i = expert_counts / num_tokens  # Fraction of tokens handled by expert i

    # P_i = average router probability assigned to expert i
    P_i = probs_flat.mean(dim=0)

    # Loss = N * sum(f_i * P_i)
    load_balance_loss = N_exp * torch.sum(f_i * P_i)
    return lambda_load * load_balance_loss


class DummyMIMICDataset(Dataset):
    """
    Dummy Dataset simulating MIMIC-IV time-series data.
    This version doesn't include RUS values - just the raw data and labels.
    """
    def __init__(self, num_samples: int, num_modalities: int, seq_len: int, input_feat_dim: int, num_classes: int):
        self.num_samples = num_samples
        self.num_modalities = num_modalities
        self.seq_len = seq_len
        self.input_feat_dim = input_feat_dim
        self.num_classes = num_classes

        # Generate random data for demonstration
        self.data = torch.randn(num_samples, num_modalities, seq_len, input_feat_dim)
        # Generate dummy labels (binary classification)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_epoch(model: BaselineMoEModel,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                task_criterion: nn.Module,
                device: torch.device,
                args: argparse.Namespace):
    """Runs one training epoch."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1}/{args.epochs} [Train]", leave=False)
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        final_logits, all_aux_moe_outputs = model(data)

        # Calculate Task Loss
        task_loss = task_criterion(final_logits, labels)

        # Calculate Load Balancing Loss (sum over all MoE layers)
        total_L_load = torch.tensor(0.0, device=device)
        num_moe_layers_encountered = 0

        # Iterate through model layers to find MoE blocks and get aux outputs
        moe_aux_output_index = 0
        for layer in model.layers:
            if isinstance(layer, BaselineMoEBlock):
                if moe_aux_output_index < len(all_aux_moe_outputs):
                    aux_outputs = all_aux_moe_outputs[moe_aux_output_index]
                    moe_aux_output_index += 1
                else:
                    print(f"Warning: Mismatch in aux outputs at batch {batch_idx}")
                    continue

                num_moe_layers_encountered += 1
                gating_probs = aux_outputs['gating_probs']
                expert_indices = aux_outputs['expert_indices']
                k = layer.moe_layer.k

                L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, args.lambda_load)
                total_L_load += L_load

        # Average load balancing loss over the number of MoE layers
        if num_moe_layers_encountered > 0:
            total_L_load /= num_moe_layers_encountered

        # Combine Losses (only task loss + load balancing loss)
        total_loss = task_loss + total_L_load

        # Backward pass and optimize
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN or Inf detected in total loss at batch {batch_idx}. Skipping backward pass.")
        else:
            total_loss.backward()
            # Optional: Gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            # Accumulate losses for logging
            total_loss_accum += total_loss.item()
            task_loss_accum += task_loss.item()
            load_loss_accum += total_L_load.item()

            # Calculate accuracy
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'TaskL': f"{task_loss.item():.4f}",
                'Acc': f"{100. * correct_predictions / total_samples:.2f}%"
            })

    # Calculate average losses and accuracy for the epoch
    avg_total_loss = total_loss_accum / len(dataloader)
    avg_task_loss = task_loss_accum / len(dataloader)
    avg_load_loss = load_loss_accum / len(dataloader)
    accuracy = 100. * correct_predictions / total_samples

    print(f"Epoch {args.current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Load Balancing Loss: {avg_load_loss:.4f}")

    return avg_total_loss, accuracy


def validate_epoch(model: BaselineMoEModel,
                   dataloader: DataLoader,
                   task_criterion: nn.Module,
                   device: torch.device,
                   args: argparse.Namespace):
    """Runs one validation epoch."""
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1}/{args.epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            final_logits, all_aux_moe_outputs = model(data)

            # Calculate Task Loss
            task_loss = task_criterion(final_logits, labels)

            # Accumulate losses for logging
            task_loss_accum += task_loss.item()

            # Calculate accuracy
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix({
                'Val TaskL': f"{task_loss.item():.4f}",
                'Val Acc': f"{100. * correct_predictions / total_samples:.2f}%"
            })

    # Calculate average losses and accuracy for the epoch
    avg_task_loss = task_loss_accum / len(dataloader)
    accuracy = 100. * correct_predictions / total_samples

    print(f"Epoch {args.current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy


def main(args):
    """Main function to set up and run training."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Seed for reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Dummy Data Loaders ---
    print("Setting up dummy datasets and dataloaders...")
    train_dataset = DummyMIMICDataset(
        num_samples=100,  # Small number for demo
        num_modalities=args.num_modalities,
        seq_len=args.seq_len,
        input_feat_dim=args.input_dim,
        num_classes=args.num_classes
    )
    val_dataset = DummyMIMICDataset(
        num_samples=20,  # Small number for demo
        num_modalities=args.num_modalities,
        seq_len=args.seq_len,
        input_feat_dim=args.input_dim,
        num_classes=args.num_classes
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Dataloaders created.")

    # --- Model Configuration ---
    moe_layer_config = {
        "input_dim": args.d_model,
        "output_dim": args.d_model,
        "num_experts": args.moe_num_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "dropout_rate": args.dropout,
    }

    print("Initializing baseline MoE model...")
    model = BaselineMoEModel(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_modalities=args.num_modalities,
        num_classes=args.num_classes,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, task_criterion, device, args)
        val_loss, val_acc = validate_epoch(model, val_loader, task_criterion, device, args)

        # Simple best model saving based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%.")

    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Baseline MoE Model on MIMIC-IV (Conceptual)')

    # Data args
    parser.add_argument('--num_modalities', type=int, default=10, help='Number of variables treated as modalities')
    parser.add_argument('--seq_len', type=int, default=48, help='Sequence length of time-series data (e.g., hours)')
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension per modality/variable')
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Maximum sequence length for positional encoding')

    # Model args
    parser.add_argument('--d_model', type=int, default=128, help='Internal dimension of the model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed-forward layer in Transformer blocks')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Total number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of layers to replace with MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes for the task')

    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Hidden dimension within experts')

    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')

    # Loss args
    parser.add_argument('--lambda_load', type=float, default=0.01, help='Weight for load balancing loss')

    args = parser.parse_args()
    args.current_epoch = 0
    main(args)
