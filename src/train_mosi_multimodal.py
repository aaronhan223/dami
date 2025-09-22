import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
from typing import Dict, List
from datetime import datetime
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from trus_moe_multimodal import MultimodalTRUSMoEModel
from trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
from plot_expert_activation import analyze_expert_activations
from multibench_affect_get_data import get_dataloader

def load_mosimosei_rus_data(rus_filepath: str, modality_names: List[str], seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Loads MOSI/MOSEI RUS data with multi-lag support and interpolation.
    """
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")
    
    print(f"Loading RUS data from: {rus_filepath}")
    all_pid_results = np.load(rus_filepath, allow_pickle=True)
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}
    
    # Initialize RUS tensors with time dimension
    T = seq_len
    U = torch.zeros(num_modalities, T, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    
    print(f"Expected modality names: {modality_names}")
    print(f"Sequence length: {seq_len}")
    print(f"Total RUS results to process: {len(all_pid_results)}")
    
    pairs_loaded = 0
    pairs_skipped_not_found = 0
    pairs_skipped_same_modality = 0
    
    for result in all_pid_results:
        mod1, mod2 = result['feature_pair']
        
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because one or both modalities not found.")
            pairs_skipped_not_found += 1
            continue
        
        m1_idx = modality_to_idx[mod1]
        m2_idx = modality_to_idx[mod2]

        if m1_idx == m2_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because they are the same modality.")
            pairs_skipped_same_modality += 1
            continue
        
        lag_results = result['lag_results']
        num_lags = len(lag_results)
        
        # Extract lag values and RUS values
        lag_values = [lag_result['lag'] for lag_result in lag_results]
        R_values = [lag_result['R_value'] for lag_result in lag_results]
        S_values = [lag_result['S_value'] for lag_result in lag_results]
        U1_values = [lag_result['U1_value'] for lag_result in lag_results]
        U2_values = [lag_result['U2_value'] for lag_result in lag_results]
        
        print(f"Loading pair ({mod1}, {mod2}) with {num_lags} lags: {lag_values}")
        
        # Map lag values directly to time indices (ensure within [0, T-1] range)
        lag_times = [min(lag, T - 1) for lag in lag_values]
        
        # Helper function for interpolation
        def interpolate_values(values):
            if len(lag_times) == 1:
                return torch.full((T,), values[0])
            else:
                full_times = torch.arange(T, dtype=torch.float32)
                return torch.from_numpy(np.interp(full_times.numpy(), lag_times, values))
        
        # Interpolate R, S, U1, U2 values
        R_interp = interpolate_values(R_values)
        S_interp = interpolate_values(S_values)
        U1_interp = interpolate_values(U1_values)
        U2_interp = interpolate_values(U2_values)
        
        # Store interpolated values
        R[m1_idx, m2_idx, :] = R_interp
        R[m2_idx, m1_idx, :] = R_interp
        S[m1_idx, m2_idx, :] = S_interp
        S[m2_idx, m1_idx, :] = S_interp
        
        # For uniqueness, take maximum across all pairs (modality can appear in multiple pairs)
        U[m1_idx, :] = torch.maximum(U[m1_idx, :], U1_interp)
        U[m2_idx, :] = torch.maximum(U[m2_idx, :], U2_interp)
        
        pairs_loaded += 1
    
    print(f"RUS data loading complete:")
    print(f"  RUS tensor shapes: U={U.shape}, R={R.shape}, S={S.shape}")
    print(f"  Average R value: {R.mean().item():.4f}")
    print(f"  Average S value: {S.mean().item():.4f}")
    print(f"  Average U value: {U.mean().item():.4f}")
    print(f"  Pairs loaded: {pairs_loaded}, Skipped (not found): {pairs_skipped_not_found}, Skipped (same): {pairs_skipped_same_modality}")
 
    return {'U': U, 'R': R, 'S': S}


def train_epoch_multimodal(model, dataloader, rus_data, optimizer, task_criterion, device, args, current_epoch):
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    unique_loss_accum = 0.0
    redundancy_loss_accum = 0.0
    synergy_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Train]", leave=False)

    for batch_idx, (vision, audio, text, labels) in enumerate(progress_bar):
        batch_size = vision.shape[0]
        modality_data = [vision.to(device), audio.to(device), text.to(device)]
        rus_values = {
            'U': torch.stack([rus_data['U'] for _ in range(batch_size)]).to(device),
            'R': torch.stack([rus_data['R'] for _ in range(batch_size)]).to(device),
            'S': torch.stack([rus_data['S'] for _ in range(batch_size)]).to(device)
        }
        labels = labels.squeeze(-1).to(device)

        # print(f"Batch {batch_idx} size: {batch_size}")
        # print(f"RUS values shape: {rus_values['U'].shape}")
        # print(f"RUS values shape: {rus_values['R'].shape}")
        # print(f"RUS values shape: {rus_values['S'].shape}")
        # print(f"Labels shape: {labels.shape}")
        # print(f"Modality data shape: {modality_data[0].shape}")
        # print(f"Modality data shape: {modality_data[1].shape}")
        # print(f"Modality data shape: {modality_data[2].shape}")

        optimizer.zero_grad()

        # Forward pass
        final_logits, all_aux_moe_outputs = model(modality_data, rus_values)

        # Calculate Task Loss
        task_loss = task_criterion(final_logits, labels)

        # Calculate Auxiliary Losses
        total_L_unique = torch.tensor(0.0, device=device)
        total_L_redundancy = torch.tensor(0.0, device=device)
        total_L_synergy = torch.tensor(0.0, device=device)
        total_L_load = torch.tensor(0.0, device=device)
        num_moe_layers = len(all_aux_moe_outputs)

        for aux_outputs in all_aux_moe_outputs:
            gating_probs = aux_outputs['gating_probs']  # (B, M, T, N_exp)
            expert_indices = aux_outputs['expert_indices']  # (B, T, k)
            
            # Get synergy expert indices from the model
            # This is a bit tricky since we need to access the actual MoE layer
            # For now, we'll use a fixed set based on configuration
            num_experts = gating_probs.size(-1)
            synergy_expert_indices = set(range(args.moe_num_synergy_experts))
            
            # Calculate RUS losses
            L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                gating_probs, rus_values, synergy_expert_indices,
                args.threshold_u, args.threshold_r, args.threshold_s,
                args.lambda_u, args.lambda_r, args.lambda_s,
                epsilon=args.epsilon_loss
            )
            
            # Calculate load balancing loss
            k = args.moe_k
            L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, args.lambda_load)
            
            total_L_unique += L_unique
            total_L_redundancy += L_redundancy
            total_L_synergy += L_synergy
            total_L_load += L_load
        
        if num_moe_layers > 0:
            total_L_unique /= num_moe_layers
            total_L_redundancy /= num_moe_layers
            total_L_synergy /= num_moe_layers
            total_L_load /= num_moe_layers
        
        # Combine Losses
        total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load
        
        total_loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        optimizer.step()
        
        # Accumulate losses and accuracy
        total_loss_accum += total_loss.item()
        task_loss_accum += task_loss.item()
        unique_loss_accum += total_L_unique.item()
        redundancy_loss_accum += total_L_redundancy.item()
        synergy_loss_accum += total_L_synergy.item()
        load_loss_accum += total_L_load.item()

        predictions = torch.argmax(final_logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar
        if total_samples > 0:
            current_acc = 100. * correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'TaskL': f"{task_loss.item():.4f}",
                'Acc': f"{current_acc:.2f}%"
            })
    
    # Calculate average losses and accuracy
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_total_loss = total_loss_accum / num_batches
    avg_task_loss = task_loss_accum / num_batches
    avg_unique_loss = unique_loss_accum / num_batches
    avg_redundancy_loss = redundancy_loss_accum / num_batches
    avg_synergy_loss = synergy_loss_accum / num_batches
    avg_load_loss = load_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    if args.use_wandb:
        wandb.log({
            "train/total_loss": avg_total_loss,
            "train/task_loss": avg_task_loss,
            "train/unique_loss": avg_unique_loss,
            "train/redundancy_loss": avg_redundancy_loss,
            "train/synergy_loss": avg_synergy_loss,
            "train/load_balancing_loss": avg_load_loss,
            "train/accuracy": accuracy,
            "epoch": current_epoch + 1
        })
    
    print(f"Epoch {current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, "
          f"Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Aux Losses -> Unique: {avg_unique_loss:.4f}, Redundancy: {avg_redundancy_loss:.4f}, "
          f"Synergy: {avg_synergy_loss:.4f}, Load: {avg_load_loss:.4f}")
    
    return avg_total_loss, accuracy

def eval_epoch_multimodal(model, dataloader, rus_data, task_criterion, device, args, current_epoch):
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Val]", leave=False)

    with torch.no_grad():
        for batch_idx, (vision, audio, text, labels) in enumerate(progress_bar):
            batch_size = vision.shape[0]
            modality_data = [vision.to(device), audio.to(device), text.to(device)]
            rus_values = {
                'U': torch.stack([rus_data['U'] for _ in range(batch_size)]).to(device),
                'R': torch.stack([rus_data['R'] for _ in range(batch_size)]).to(device),
                'S': torch.stack([rus_data['S'] for _ in range(batch_size)]).to(device)
            }
            labels = labels.squeeze(-1).to(device)

            # Forward pass
            final_logits, _ = model(modality_data, rus_values)

            # Calculate Task Loss
            task_loss = task_criterion(final_logits, labels)
            task_loss_accum += task_loss.item()

            # Calculate accuracy
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            if total_samples > 0:
                current_acc = 100. * correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Val TaskL': f"{task_loss.item():.4f}",
                    'Val Acc': f"{current_acc:.2f}%"
                })
    
    # Calculate average losses and accuracy
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_task_loss = task_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    if args.use_wandb:
        wandb.log({
            "val/task_loss": avg_task_loss,
            "val/accuracy": accuracy,
            "epoch": current_epoch + 1
        })
    
    print(f"Epoch {current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_task_loss, accuracy

def main(args):

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"mosi_multimodal_rus_moe_{args.dataset}_{timestamp}"

    # Set up wandb

    if args.use_wandb:        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=args.run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )


    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    if args.dataset == 'mosi':   
        modality_dim_dict = {
            'vision': 35,
            'audio': 74,
            'text': 300,
        }
    elif args.dataset == 'mosei':
        modality_dim_dict = {
            'vision': 713,
            'audio': 74,
            'text': 300,
        }
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    print(f"Modality dimensions: {modality_dim_dict}")

    print(f"Loading data...")
    modality_names = ['vision', 'audio', 'text']
    rus_data = load_mosimosei_rus_data(args.rus_data_path, modality_names, args.seq_len)
    train_loader, val_loader, _ = get_dataloader(args.dataset_path, data_type=args.dataset, max_pad=True, task='classification', max_seq_len=args.seq_len, batch_size=args.batch_size)

    
    modality_configs = []

    for mod_name in modality_names:
        config = {
            'input_dim': modality_dim_dict[mod_name],
            'num_layers': args.modality_encoder_layers,
            'nhead': args.nhead,
            'd_ff': args.d_ff,
            'use_cnn': args.use_cnn_encoders,
            'kernel_size': 3
        }
        modality_configs.append(config)

    # MoE configuration
    moe_router_config = {
        "gru_hidden_dim": args.moe_router_gru_hidden_dim,
        "token_processed_dim": args.moe_router_token_processed_dim,
        "attn_key_dim": args.moe_router_attn_key_dim,
        "attn_value_dim": args.moe_router_attn_value_dim,
    }
    
    moe_layer_config = {
        "num_experts": args.moe_num_experts,
        "num_synergy_experts": args.moe_num_synergy_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "synergy_expert_nhead": args.nhead,
        "router_config": moe_router_config,
        "capacity_factor": args.moe_capacity_factor,
        "drop_tokens": args.moe_drop_tokens,
    }

    print(f"Device: {device}")

    model = MultimodalTRUSMoEModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_classes=2,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_checkpoint=args.use_gradient_checkpointing,
        output_attention=False
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    os.makedirs(os.path.join(args.output_dir, 'checkpoints', args.run_name), exist_ok=True)

    task_criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_multimodal(
            model, train_loader, rus_data, optimizer, task_criterion, device, args, epoch
        )

        val_loss, val_acc = eval_epoch_multimodal(
            model, val_loader, rus_data, task_criterion, device, args, epoch
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            save_path = os.path.join(args.output_dir, 'checkpoints', args.run_name, f'best_multimodal_model_{args.dataset}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'args': args,
                'modality_configs': modality_configs,
                'modality_names': modality_names,
            }, save_path)
            print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved.")

            if args.use_wandb:
                wandb.log({"best_val_acc": best_val_acc, "best_epoch": epoch + 1})

        if scheduler is not None:
            scheduler.step()

    print("Training finished.")
    if best_epoch != -1:
        print(f"Best Validation Accuracy: {best_val_acc:.4f}% at epoch {best_epoch+1}")

        if args.plot_expert_activations and len(val_loader) > 0:
            print("\nGenerating expert activation plots for the best multimodal TRUS-MoE model...")
            
            best_model_path = os.path.join(args.output_dir, 'checkpoints', args.run_name, f'best_multimodal_model_{args.dataset}.pth')
            if os.path.exists(best_model_path):
                # Add argparse.Namespace to safe globals for PyTorch 2.6+ compatibility
                torch.serialization.add_safe_globals([argparse.Namespace])
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                
                # Create a fresh model instance
                plot_model = MultimodalTRUSMoEModel(
                    modality_configs=modality_configs,
                    d_model=args.d_model,
                    nhead=args.nhead,
                    d_ff=args.d_ff,
                    num_encoder_layers=args.num_encoder_layers,
                    num_moe_layers=args.num_moe_layers,
                    moe_config=moe_layer_config,
                    num_classes=2,
                    max_seq_len=args.seq_len,
                    dropout=args.dropout,
                    use_checkpoint=args.use_gradient_checkpointing,
                    output_attention=False
                ).to(device)

                plot_model.load_state_dict(checkpoint['model_state_dict'])
                plot_model.eval()
                
                # Get a batch of validation data
                plot_loader = get_dataloader(args.dataset_path, data_type=args.dataset, max_pad=True, task='classification', max_seq_len=args.seq_len, batch_size=args.plot_num_samples)

                plot_iter = iter(plot_loader)
                batch_modalities, batch_masks, batch_rus, batch_labels = next(plot_iter)
                # Move to device (same as training/validation loops)
                batch_modalities = [mod.to(device) for mod in batch_modalities]
                batch_rus = {k: v.to(device) for k, v in batch_rus.items()}
                
                # Generate plots
                plot_save_dir = os.path.join(args.output_dir, 'expert_activation_plots', args.run_name)

                try:
                    analyze_expert_activations(
                        trus_model=plot_model,
                        baseline_model=None,
                        data_batch=batch_modalities,
                        rus_values=batch_rus,
                        modality_names=modality_names,
                        save_dir=plot_save_dir
                    )
                    print(f"Expert activation plots saved to {plot_save_dir}")
                except Exception as e:
                    print(f"Error generating expert activation plots: {e}")

            else:
                print(f"Best model checkpoint not found at {best_model_path}")

        else:
            print("Training finished (no validation performed or no improvement).")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
                
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multimodal TRUS-MoE Model on MOSI/MOSEI Data')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--rus_data_path', type=str, required=True, help='Path to the RUS data')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to train on')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length')

    # Model architecture args
    parser.add_argument('--d_model', type=int, default=32, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=64, help='Feed-forward dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--modality_encoder_layers', type=int, default=2, 
                       help='Number of layers in modality-specific encoders')
    parser.add_argument('--use_cnn_encoders', action='store_true', 
                       help='Use CNN layers in modality encoders')
    
    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_num_synergy_experts', type=int, default=2, help='Number of synergy experts')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Expert hidden dimension')
    parser.add_argument('--moe_capacity_factor', type=float, default=1.25, help='Expert capacity factor')
    parser.add_argument('--moe_drop_tokens', action='store_true', help='Drop tokens exceeding capacity')
    
    # MoE router args
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=32, help='GRU hidden dim in router')
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=32, 
                       help='Token processing dim in router')
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=16, help='Attention key dim in router')
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=16, 
                       help='Attention value dim in router')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, 
                       help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use cosine annealing LR scheduler')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', 
                       help='Use gradient checkpointing to save memory')
    
    # Loss args
    parser.add_argument('--threshold_u', type=float, default=0.01, help='Threshold for uniqueness loss')
    parser.add_argument('--threshold_r', type=float, default=0.01, help='Threshold for redundancy loss')
    parser.add_argument('--threshold_s', type=float, default=0.01, help='Threshold for synergy loss')
    parser.add_argument('--lambda_u', type=float, default=1, help='Weight for uniqueness loss')
    parser.add_argument('--lambda_r', type=float, default=1, help='Weight for redundancy loss')
    parser.add_argument('--lambda_s', type=float, default=1, help='Weight for synergy loss')
    parser.add_argument('--lambda_load', type=float, default=0.02, help='Weight for load balancing loss')
    parser.add_argument('--epsilon_loss', type=float, default=1e-8, help='Epsilon for loss stability')

    # System args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use. If None, will use CPUs.')
    parser.add_argument('--output_dir', type=str, default='../results/affect/', 
                       help='Directory to save results/models')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mosi-multimodal-trus-moe', 
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--run_name', type=str, default=None, help='run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')

    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true', help='Generate expert activation plots after training')
    parser.add_argument('--plot_num_samples', type=int, default=32, help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    main(args)