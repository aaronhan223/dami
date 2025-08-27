import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import random
from typing import Dict, Tuple, List, Optional
import numpy as np
from tqdm import tqdm

# Import necessary components from the original TRUS MoE implementation
from trus_moe_model import (
    PositionalEncoding, FeedForwardExpert, TransformerBlock
)


class VanillaRouter(nn.Module):
    """
    Vanilla router with standard top-k softmax gating mechanism.
    No RUS information is used, just regular token-level routing.
    """
    def __init__(self, d_model: int, num_experts: int, noise_std: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.noise_std = noise_std
        
        # Simple linear layer for routing
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model) or (B, M, T, d_model)
        Returns:
            router_logits: Routing logits of same shape as input but with num_experts in last dim
        """
        # Add noise during training for load balancing
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            
        # Apply gate to get routing logits
        router_logits = self.gate(x)  # (..., num_experts)
        return router_logits


class ModalitySpecificEncoder(nn.Module):
    """
    Modality-specific encoder that processes raw features from a single modality.
    Each modality can have different input dimensions and processing architectures.
    """
    def __init__(self, 
                 modality_input_dim: int,
                 d_model: int,
                 num_layers: int = 2,
                 nhead: int = 4,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 use_cnn: bool = False,
                 kernel_size: int = 3):
        super().__init__()
        
        self.modality_input_dim = modality_input_dim
        self.d_model = d_model
        self.use_cnn = use_cnn
        
        # Input projection
        self.input_proj = nn.Linear(modality_input_dim, d_model)
        
        # Optional CNN layers for temporal feature extraction
        if use_cnn:
            self.cnn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(2)
            ])
        
        # Transformer layers for modality-specific encoding
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D_in) where D_in is modality_input_dim
        Returns:
            Encoded features of shape (B, T, d_model)
        """
        B, T, D_in = x.shape
        assert D_in == self.modality_input_dim, f"Expected input dim {self.modality_input_dim}, got {D_in}"
        
        # Project to d_model
        x = self.input_proj(x) * math.sqrt(self.d_model)  # (B, T, d_model)
        
        # Apply CNN if enabled
        if self.use_cnn:
            # Transpose for CNN: (B, T, d_model) -> (B, d_model, T)
            x_cnn = x.transpose(1, 2)
            for cnn_layer in self.cnn_layers:
                x_cnn = x_cnn + cnn_layer(x_cnn)  # Residual connection
            # Transpose back: (B, d_model, T) -> (B, T, d_model)
            x = x_cnn.transpose(1, 2)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        x = self.norm(x)
        return x


def load_balancing_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute load balancing loss to encourage uniform distribution of tokens to experts.
    
    Args:
        router_probs: Router probabilities of shape (B*M*T, num_experts)
        expert_indices: Selected expert indices of shape (B*M*T, k)
        num_experts: Total number of experts
    Returns:
        load_balancing_loss: Scalar loss value
    """
    # Compute fraction of tokens assigned to each expert
    expert_counts = torch.zeros(num_experts, device=router_probs.device)
    for i in range(num_experts):
        expert_counts[i] = (expert_indices == i).sum().float()
    
    # Normalize to get fractions
    total_tokens = expert_indices.numel()
    expert_fractions = expert_counts / total_tokens
    
    # Compute average router probability for each expert
    mean_router_probs = router_probs.mean(dim=0)
    
    # Load balancing loss: encourage uniform distribution
    # Loss = coefficient * num_experts * sum(f_i * P_i) where f_i is fraction and P_i is mean prob
    load_balance_loss = num_experts * torch.sum(expert_fractions * mean_router_probs)
    
    return load_balance_loss


class MultimodalBaselineMoELayer(nn.Module):
    """
    Baseline multimodal MoE layer with vanilla router and load balancing loss.
    Uses regular feedforward experts without RUS information.
    """
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 k: int,
                 expert_hidden_dim: int,
                 router_noise_std: float = 0.1,
                 load_balance_weight: float = 0.01,
                 use_load_balancing: bool = True,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.load_balance_weight = load_balance_weight
        self.use_load_balancing = use_load_balancing
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Create regular feedforward experts only
        self.experts = nn.ModuleList([
            FeedForwardExpert(d_model, expert_hidden_dim, d_model)
            for _ in range(num_experts)
        ])
        
        # Vanilla router
        self.router = VanillaRouter(d_model, num_experts, router_noise_std)
        
    def forward(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            modality_features: List of tensors, each of shape (B, T, d_model)
        Returns:
            output: (B, T, d_model) - fused output across modalities
            aux_outputs: Dictionary with auxiliary information
        """
        B = modality_features[0].size(0)
        T = modality_features[0].size(1)
        M = len(modality_features)
        device = modality_features[0].device
        
        # Stack modality features: List[(B, T, d_model)] -> (B, M, T, d_model)
        x_stacked = torch.stack(modality_features, dim=1)  # (B, M, T, d_model)
        
        # Get routing decisions for each modality independently
        router_logits = self.router(x_stacked)  # (B, M, T, num_experts)
        
        # Flatten for processing: (B, M, T, num_experts) -> (B*M*T, num_experts)
        router_logits_flat = router_logits.view(B * M * T, self.num_experts)
        gating_probs_flat = F.softmax(router_logits_flat, dim=-1)
        
        # Top-k routing for each modality-time position independently
        top_k_weights, top_k_indices = torch.topk(router_logits_flat, self.k, dim=-1)
        top_k_probs = F.softmax(top_k_weights, dim=-1)
        
        # Process tokens through experts
        num_tokens = B * M * T
        x_flat = x_stacked.view(num_tokens, self.d_model)
        output_flat = torch.zeros_like(x_flat)
        
        # Efficient batched expert processing
        flat_expert_indices = top_k_indices.view(-1)
        flat_router_probs = top_k_probs.view(-1)
        flat_batch_indices = torch.arange(num_tokens, device=device).repeat_interleave(self.k)
        
        for expert_idx in range(self.num_experts):
            mask = (flat_expert_indices == expert_idx)
            if mask.any():
                original_token_indices = flat_batch_indices[mask]
                current_routing_probs = flat_router_probs[mask].unsqueeze(-1)
                expert_input = x_flat[original_token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_expert_output = expert_output * current_routing_probs
                output_flat.index_add_(0, original_token_indices, weighted_expert_output)
        
        # Reshape back to (B, M, T, d_model)
        output_modalities = output_flat.view(B, M, T, self.d_model)
        
        # Fuse modalities by averaging (simple fusion strategy)
        fused_output = output_modalities.mean(dim=1)  # (B, T, d_model)
        
        # Compute load balancing loss
        load_balance_loss = torch.tensor(0.0, device=device)
        if self.use_load_balancing and self.training:
            load_balance_loss = load_balancing_loss(gating_probs_flat, top_k_indices.view(-1, self.k), self.num_experts)
        
        # Prepare auxiliary outputs
        aux_outputs = {
            "gating_probs": gating_probs_flat.view(B, M, T, self.num_experts),
            "expert_indices": top_k_indices.view(B, M, T, self.k),
            "router_logits": router_logits,
            "modality_outputs": output_modalities,
            "load_balance_loss": load_balance_loss,
        }
        
        return fused_output, aux_outputs


class MultimodalBaselineMoEBlock(nn.Module):
    """
    Transformer block with multimodal baseline MoE layer.
    Handles multiple modality streams independently.
    """
    def __init__(self, d_model: int, nhead: int, 
                 moe_layer: MultimodalBaselineMoELayer, 
                 dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.moe_layer = moe_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_moe = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint
        
        # Self-attention for each modality independently
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(4)  # Default to 4 modalities, will adjust during forward
        ])
        self.dropout1 = nn.Dropout(dropout)
        
    def _forward(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if modality_features is None or len(modality_features) == 0:
            raise ValueError("modality_features cannot be None or empty")
            
        num_modalities = len(modality_features)
        
        # Apply self-attention to each modality independently
        attended_features = []
        for i, features in enumerate(modality_features):
            if i < len(self.self_attns):
                attn_module = self.self_attns[i]
            else:
                # Create new attention module if needed
                attn_module = nn.MultiheadAttention(features.size(-1), 4, batch_first=True).to(features.device)
            
            attn_output, _ = attn_module(features, features, features, need_weights=False)
            attended = features + self.dropout1(attn_output)
            attended = self.norm1(attended)
            attended_features.append(attended)
        
        # MoE layer processes all modalities and returns fused output
        moe_output, aux_moe_outputs = self.moe_layer(attended_features)
        
        # Add residual connection with the mean of input modalities
        input_mean = torch.stack(modality_features, dim=1).mean(dim=1)  # (B, T, d_model)
        output = input_mean + self.dropout_moe(moe_output)
        output = self.norm2(output)
        
        return output, aux_moe_outputs
    
    def forward(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # Ensure we have the right number of attention modules
        if len(self.self_attns) != len(modality_features):
            # Adjust the number of attention modules
            while len(self.self_attns) < len(modality_features):
                d_model = modality_features[0].size(-1)
                nhead = 4  # Default value
                new_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True).to(modality_features[0].device)
                self.self_attns.append(new_attn)
        
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory
            output, aux_outputs = checkpoint(self._forward, modality_features, use_reentrant=False)
            return output, aux_outputs
        else:
            return self._forward(modality_features)


class MultimodalBaselineMoEModel(nn.Module):
    """
    Baseline multimodal MoE model with vanilla router and load balancing loss.
    Uses regular feedforward experts without RUS information.
    """
    def __init__(self,
                 modality_configs: List[Dict],  # List of configs for each modality
                 d_model: int,
                 nhead: int,
                 d_ff: int,
                 num_encoder_layers: int,
                 num_moe_layers: int,
                 moe_config: Dict,
                 num_classes: int,
                 max_seq_len: int = 1000,
                 dropout: float = 0.1,
                 use_checkpoint: bool = False,
                 output_attention: bool = False):
        super().__init__()
        
        self.num_modalities = len(modality_configs)
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.output_attention = output_attention
        
        # Create modality-specific encoders
        self.modality_encoders = nn.ModuleList()
        for config in modality_configs:
            encoder = ModalitySpecificEncoder(
                modality_input_dim=config['input_dim'],
                d_model=d_model,
                num_layers=config.get('num_layers', 2),
                nhead=config.get('nhead', nhead),
                d_ff=config.get('d_ff', d_ff),
                dropout=dropout,
                use_cnn=config.get('use_cnn', False),
                kernel_size=config.get('kernel_size', 3)
            )
            self.modality_encoders.append(encoder)
        
        # Positional encoding (applied to each modality independently)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList()
        
        # Determine which layers will be MoE layers
        moe_indices = set()
        if num_moe_layers > 0 and num_encoder_layers > 0:
            step = max(1, num_encoder_layers // num_moe_layers)
            for i in range(num_moe_layers):
                idx = min(i * step, num_encoder_layers - 1)
                moe_indices.add(idx)
        
        print(f"Baseline MoE layers at indices: {sorted(list(moe_indices))}")
        
        for i in range(num_encoder_layers):
            if i in moe_indices:
                moe_layer = MultimodalBaselineMoELayer(
                    d_model=d_model,
                    **moe_config
                )
                layer = MultimodalBaselineMoEBlock(
                    d_model, nhead, moe_layer, dropout, use_checkpoint
                )
            else:
                # For non-MoE layers, use regular transformer
                layer = TransformerBlock(d_model, nhead, d_ff, dropout)
            self.layers.append(layer)
        
        # Output layers
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)
    
    def forward(self, 
                modality_inputs: List[torch.Tensor],
                return_embeddings: bool = False) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Args:
            modality_inputs: List of tensors, each of shape (B, T, D_m) where D_m 
                           is the input dimension for modality m
            return_embeddings: If True, return intermediate embeddings
        Returns:
            final_logits: (B, num_classes)
            all_aux_outputs: List of auxiliary outputs from MoE layers
        """
        assert len(modality_inputs) == self.num_modalities
        B = modality_inputs[0].size(0)
        T = modality_inputs[0].size(1)
        
        # Step 1: Encode each modality
        encoded_modalities = []
        for i, (encoder, mod_input) in enumerate(zip(self.modality_encoders, modality_inputs)):
            encoded = encoder(mod_input)  # (B, T, d_model)
            
            # Add positional encoding to each modality
            # Reshape for positional encoding: (B, T, d_model) -> (T, B, d_model)
            encoded_pos = encoded.transpose(0, 1)
            encoded_pos = self.pos_encoder(encoded_pos)
            encoded_pos = encoded_pos.transpose(0, 1)  # Back to (B, T, d_model)
            
            encoded_modalities.append(encoded_pos)
        
        # Step 2: Pass through transformer layers
        all_aux_outputs = []
        intermediate_embeddings = []
        current_modalities = encoded_modalities
        
        for layer in self.layers:
            if isinstance(layer, MultimodalBaselineMoEBlock):
                # MoE layer: processes multiple modalities and returns fused output
                fused_output, aux_outputs = layer(current_modalities)
                all_aux_outputs.append(aux_outputs)
                
                # For subsequent layers, we have a fused representation
                # Convert back to modality list for consistency
                current_modalities = [fused_output] * self.num_modalities
                
            elif isinstance(layer, TransformerBlock):
                # Regular transformer: apply to each modality independently
                processed_modalities = []
                for modality in current_modalities:
                    processed = layer(modality)
                    processed_modalities.append(processed)
                current_modalities = processed_modalities
            
            if return_embeddings:
                if isinstance(layer, MultimodalBaselineMoEBlock):
                    intermediate_embeddings.append(fused_output.clone())
                else:
                    intermediate_embeddings.append(torch.stack(current_modalities, dim=1).mean(dim=1).clone())
        
        # Step 3: Final processing
        # If we still have multiple modalities, fuse them
        if len(current_modalities) > 1:
            final_features = torch.stack(current_modalities, dim=1).mean(dim=1)  # (B, T, d_model)
        else:
            final_features = current_modalities[0]
        
        final_features = self.final_norm(final_features)
        
        # Global average pooling
        pooled_output = final_features.mean(dim=1)  # (B, d_model)
        
        # Final classification
        final_logits = self.output_proj(pooled_output)  # (B, num_classes)
        
        # Package outputs
        outputs = {
            'logits': final_logits,
            'aux_moe_outputs': all_aux_outputs,
        }
        
        if return_embeddings:
            outputs['embeddings'] = intermediate_embeddings
            outputs['encoded_modalities'] = encoded_modalities
        
        return final_logits, all_aux_outputs


def compute_total_loss(logits: torch.Tensor, 
                      targets: torch.Tensor, 
                      aux_outputs: List[Dict],
                      load_balance_weight: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Compute total loss including classification loss and load balancing loss.
    
    Args:
        logits: Model predictions of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
        aux_outputs: List of auxiliary outputs from MoE layers
        load_balance_weight: Weight for load balancing loss
    Returns:
        Dictionary with different loss components
    """
    # Classification loss
    classification_loss = F.cross_entropy(logits, targets)
    
    # Load balancing loss
    total_load_balance_loss = torch.tensor(0.0, device=logits.device)
    if aux_outputs:
        for aux_output in aux_outputs:
            if 'load_balance_loss' in aux_output:
                total_load_balance_loss += aux_output['load_balance_loss']
    
    # Total loss
    total_loss = classification_loss + load_balance_weight * total_load_balance_loss
    
    return {
        'total_loss': total_loss,
        'classification_loss': classification_loss,
        'load_balance_loss': total_load_balance_loss,
    }