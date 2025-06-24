import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import math
import os
import sys
import argparse
import random
import wandb
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Set
import pdb

try:
    from baseline_moe_model import BaselineMoEModel, BaselineMoEBlock, calculate_load_balancing_loss
    from pamap_rus import get_pamap_column_names, load_pamap_data, preprocess_pamap_data
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Please ensure baseline_moe_model.py and pamap_rus.py are accessible.")
    sys.exit(1)


DEFAULT_DATASET_DIR = "/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol"
DEFAULT_OUTPUT_DIR = "../results/pamap_training_baseline"


class PamapWindowDatasetBaseline(Dataset):
    """
    PyTorch Dataset for PAMAP2 activity recognition using sliding windows.
    This version is for baseline models that don't use RUS information.
    """
    def __init__(self, subject_id: int, data_dir: str, sensor_columns: List[str], 
                 seq_len: int, step: int, activity_map: Dict[int, int]):
        """
        Args:
            subject_id: The subject ID to load data for.
            data_dir: Directory containing PAMAP2 .dat files.
            sensor_columns: List of sensor columns to use as modalities.
            seq_len: Length of the sliding window (T).
            step: Step size for the sliding window.
            activity_map: Dictionary mapping original activity IDs to 0-based indices.
        """
        self.subject_id = subject_id
        self.data_dir = data_dir
        self.sensor_columns = sensor_columns
        self.seq_len = seq_len
        self.step = step
        self.activity_map = activity_map
        self.num_modalities = len(sensor_columns) # M
        self.input_dim = 1 # Each sensor value is a single feature

        try:
            df = load_pamap_data(self.subject_id, self.data_dir)
            # Use only selected sensor columns + activity_id
            selected_cols_with_id = ['timestamp', 'activity_id'] + self.sensor_columns
            # Ensure columns exist in the loaded dataframe
            cols_to_use = [col for col in selected_cols_with_id if col in df.columns]
            missing_cols = set(selected_cols_with_id) - set(cols_to_use)
            if missing_cols:
                print(f"Warning: Requested sensor columns not found in data: {missing_cols}")

            df_subset = df[cols_to_use].copy()
            self.df_processed, _ = preprocess_pamap_data(df_subset) # Reuse preprocessing

            # Remap activity IDs to 0-based indices
            self.df_processed['activity_label'] = self.df_processed['activity_id'].map(activity_map)
            # Drop rows where activity mapping failed (NaN label) or original activity was 0
            self.df_processed.dropna(subset=['activity_label'], inplace=True)
            self.df_processed['activity_label'] = self.df_processed['activity_label'].astype(int)

        except FileNotFoundError as e:
            print(f"Error loading data for subject {subject_id}: {e}")
            raise
        except Exception as e:
            print(f"Error processing data for subject {subject_id}: {e}")
            raise # Re-raise other unexpected errors

        if self.df_processed.empty:
             print(f"Warning: No data remaining for subject {subject_id} after preprocessing and activity mapping.")
             self.windows = []
             self.labels = []
        else:
             self._create_windows()

    def _create_windows(self):
        """Creates sliding windows from the processed dataframe."""
        self.windows = []
        self.labels = []
        data_values = self.df_processed[self.sensor_columns].values # (Total_Samples, M)
        label_values = self.df_processed['activity_label'].values # (Total_Samples,)

        total_samples = len(data_values)
        for i in range(0, total_samples - self.seq_len + 1, self.step):
            window_data = data_values[i : i + self.seq_len] # (T, M)
            window_label_codes = label_values[i : i + self.seq_len] # (T,)

            # Assign label based on the most frequent activity in the window
            unique_labels, counts = np.unique(window_label_codes, return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]

            # Reshape data: (T, M) -> (M, T, E_in=1)
            window_data_reshaped = window_data.T[:, :, np.newaxis]

            self.windows.append(torch.tensor(window_data_reshaped, dtype=torch.float32))
            self.labels.append(torch.tensor(most_frequent_label, dtype=torch.long))

        print(f"Created {len(self.windows)} windows for subject {self.subject_id}.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Data shape (M, T, E_in=1), label (scalar) - no RUS data for baseline
        return self.windows[idx], self.labels[idx]


def train_epoch(model: BaselineMoEModel,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                task_criterion: nn.Module,
                device: torch.device,
                args: argparse.Namespace,
                current_epoch: int):
    """Runs one training epoch for baseline model."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    # Set epoch for distributed sampler if using
    if args.distributed and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(current_epoch)

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Train]", leave=False, 
                         disable=args.distributed and dist.get_rank() != 0)
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        # data shape: (B, M, T, E_in)
        # labels shape: (B,)
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

        # Get the actual model (unwrap DDP if needed)
        actual_model = model.module if hasattr(model, 'module') else model

        moe_aux_output_index = 0
        for layer in actual_model.layers:
            if isinstance(layer, BaselineMoEBlock):
                if moe_aux_output_index < len(all_aux_moe_outputs):
                    aux_outputs = all_aux_moe_outputs[moe_aux_output_index]
                    moe_aux_output_index += 1
                else:
                    print(f"Warning: Mismatch in aux outputs at batch {batch_idx}")
                    continue

                num_moe_layers_encountered += 1
                gating_probs = aux_outputs['gating_probs'] # (B, S, N_exp)
                expert_indices = aux_outputs['expert_indices'] # (B, S, k)
                k = layer.moe_layer.k

                # Only load balancing loss for baseline
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
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            # Accumulate losses and accuracy
            total_loss_accum += total_loss.item()
            task_loss_accum += task_loss.item()
            load_loss_accum += total_L_load.item() if torch.is_tensor(total_L_load) else total_L_load

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

    # Calculate average losses and accuracy for the epoch
    num_batches = len(dataloader)
    if num_batches == 0: return 0.0, 0.0 # Avoid division by zero

    avg_total_loss = total_loss_accum / num_batches
    avg_task_loss = task_loss_accum / num_batches
    avg_load_loss = load_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    # Log metrics to wandb
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb.log({
            "train/total_loss": avg_total_loss,
            "train/task_loss": avg_task_loss,
            "train/load_balancing_loss": avg_load_loss,
            "train/accuracy": accuracy,
            "epoch": current_epoch + 1
        })

    print(f"Epoch {current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Load Balancing Loss: {avg_load_loss:.4f}")

    return avg_total_loss, accuracy


def validate_epoch(model: BaselineMoEModel,
                   dataloader: DataLoader,
                   task_criterion: nn.Module,
                   device: torch.device,
                   args: argparse.Namespace,
                   current_epoch: int):
    """Runs one validation epoch for baseline model."""
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            final_logits, _ = model(data) # Ignore aux outputs for validation

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

    # Calculate average losses and accuracy for the epoch
    num_batches = len(dataloader)
    if num_batches == 0: return 0.0, 0.0 # Avoid division by zero

    avg_task_loss = task_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    # Log validation metrics to wandb
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb.log({
            "val/task_loss": avg_task_loss,
            "val/accuracy": accuracy,
            "epoch": current_epoch + 1
        })

    print(f"Epoch {current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy


def main(args):
    """Main function to set up and run training."""
    # Initialize wandb if enabled - only on main process if distributed
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb_config = {k: v for k, v in vars(args).items()}
        run_name = f"baseline_subj{args.subject_id}_seq{args.seq_len}_moe{args.num_moe_layers}_lambda{args.lambda_load}"
        if args.wandb_run_name:
            run_name = args.wandb_run_name
            
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )
    
    # Set up distributed training if requested
    if args.distributed:
        if 'LOCAL_RANK' not in os.environ:
            raise ValueError("For distributed training, please launch with torchrun or python -m torch.distributed.launch")
        
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ.get('RANK', local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Set device based on local_rank
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        
        # Only print from rank 0
        is_main_process = global_rank == 0
        
        if is_main_process:
            print(f"Distributed training initialized with world_size: {world_size}")
            print(f"Using device: {device}, rank: {global_rank}/{world_size-1}")
    else:
        # Non-distributed mode
        if args.cuda_device >= 0 and torch.cuda.is_available() and not args.no_cuda:
            cuda_count = torch.cuda.device_count()
            if args.cuda_device >= cuda_count:
                print(f"Warning: Requested GPU #{args.cuda_device} but only {cuda_count} GPUs available.")
                print(f"Falling back to default CUDA device.")
                device = torch.device("cuda")
            else:
                device = torch.device(f"cuda:{args.cuda_device}")
                gpu_name = torch.cuda.get_device_name(args.cuda_device)
                print(f"Using specific CUDA device: {device} (GPU #{args.cuda_device}: {gpu_name})")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            if device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using default CUDA device: {device} ({gpu_name})")
            else:
                print(f"Using device: {device}")
        is_main_process = True

    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and Preprocess Data (Get full sensor list and activity map first) ---
    print("Loading initial data to determine sensors and activities...")
    try:
        # Load data for one subject just to get columns and activities
        temp_df = load_pamap_data(args.subject_id, args.data_dir)
        temp_df_processed, all_sensor_columns = preprocess_pamap_data(temp_df)
        # Filter sensor columns based on prefix if needed (e.g., only 'acc')
        if args.sensor_prefix:
             sensor_columns = [col for col in all_sensor_columns if col.startswith(args.sensor_prefix)]
             print(f"Selected {len(sensor_columns)} sensor columns with prefix '{args.sensor_prefix}'.")
        else:
             sensor_columns = all_sensor_columns
             print(f"Using all {len(sensor_columns)} available sensor columns.")

        if not sensor_columns:
             print("Error: No sensor columns selected. Check prefixes or data.")
             sys.exit(1)

        # Determine unique activities and create mapping
        unique_activities = sorted([act for act in temp_df_processed['activity_id'].unique() if act != 0])
        activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}
        num_classes = len(activity_map)
        print(f"Found {num_classes} activities: {unique_activities}")
        print(f"Activity mapping: {activity_map}")

        if num_classes == 0:
             print("Error: No valid activities found in the data.")
             sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}. Check --data_dir and --subject_id.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during initial data loading: {e}")
        sys.exit(1)

    # --- Create Datasets and DataLoaders ---
    if is_main_process:
        print("Creating baseline dataset...")
    try:
        full_dataset = PamapWindowDatasetBaseline(
            subject_id=args.subject_id,
            data_dir=args.data_dir,
            sensor_columns=sensor_columns,
            seq_len=args.seq_len,
            step=args.window_step,
            activity_map=activity_map
        )
    except Exception as e:
         if is_main_process:
             print(f"Error creating dataset: {e}")
         if args.distributed:
             dist.destroy_process_group()
         sys.exit(1)

    if len(full_dataset) == 0:
        if is_main_process:
            print("Error: Dataset is empty after processing and windowing.")
        if args.distributed:
            dist.destroy_process_group()
        sys.exit(1)

    # Split dataset into training and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    if train_size == 0 or val_size == 0:
         if is_main_process:
             print(f"Warning: Dataset size ({len(full_dataset)}) too small for split ({args.val_split}). Adjusting split.")
             # Adjust split, e.g., ensure at least one sample in each
             if len(full_dataset) > 1:
                 val_size = max(1, val_size)
                 train_size = len(full_dataset) - val_size
             else: # Can't split
                 train_size = len(full_dataset)
                 val_size = 0
                 print("Warning: Cannot create validation set with only 1 sample.")

    generator = torch.Generator().manual_seed(args.seed) # Ensure reproducible split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    if is_main_process:
        print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    # Create samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            seed=args.seed
        ) if val_size > 0 else None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using DistributedSampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # No shuffling for validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    ) if val_size > 0 else []
    
    if is_main_process:
        print("Dataloaders created.")

    # --- Model Configuration ---
    num_modalities = len(sensor_columns) # M
    input_dim = 1 # E_in

    if is_main_process:
        print("Initializing baseline MoE model...")
    
    # Baseline MoE model configuration
    moe_layer_config = {
        "input_dim": args.d_model,
        "output_dim": args.d_model,
        "num_experts": args.moe_num_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "dropout_rate": args.dropout,
    }
    
    model = BaselineMoEModel(
        input_dim=input_dim, # E_in = 1
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_modalities=num_modalities, # M
        num_classes=num_classes,
        dropout=args.dropout,
        max_seq_len=num_modalities * args.seq_len # S = M * T
    ).to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    if is_main_process:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_criterion = nn.CrossEntropyLoss()

    if is_main_process:
        print("Starting training...")
    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, task_criterion, device, args, epoch)

        if len(val_loader) > 0:
             val_loss, val_acc = validate_epoch(model, val_loader, task_criterion, device, args, epoch)

             # For distributed training, all processes should have same val_acc after all-reduce
             if args.distributed:
                 val_acc_tensor = torch.tensor([val_acc], device=device)
                 dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
                 val_acc = val_acc_tensor.item() / dist.get_world_size()

             # Simple best model saving based on validation accuracy (only main process saves)
             if val_acc > best_val_accuracy and is_main_process:
                 best_val_accuracy = val_acc
                 best_epoch = epoch
                 
                 # Save best model
                 save_path = os.path.join(args.output_dir, f'best_model_baseline_subj{args.subject_id}.pth')
                 
                 # Save model state dict without DDP wrapper if distributed
                 if args.distributed:
                     model_to_save = model.module
                 else:
                     model_to_save = model
                     
                 torch.save({
                     'epoch': epoch,
                     'model_state_dict': model_to_save.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'best_val_accuracy': best_val_accuracy,
                     'args': args # Save args for reference
                 }, save_path)
                 
                 print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved to {save_path}")
                 
                 # Log best model to wandb if enabled
                 if args.use_wandb:
                     wandb.log({"best_val_accuracy": best_val_accuracy, "best_epoch": epoch + 1})
                     # Save best model as an artifact
                     if args.wandb_log_model:
                         model_artifact = wandb.Artifact(f"baseline-model-subj{args.subject_id}", type="model")
                         model_artifact.add_file(save_path)
                         wandb.log_artifact(model_artifact)
        else:
             if is_main_process:
                 print(f"Epoch {epoch+1}: No validation set. Skipping validation and model saving.")

    # Synchronize processes before finishing
    if args.distributed:
        dist.barrier()

    if is_main_process:
        print("Training finished.")
        if best_epoch != -1:
            print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")
        else:
            print("Training finished (no validation performed or no improvement).")
        
    # Finish wandb run
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb.finish()
        
    # Clean up distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Baseline MoE Model on PAMAP2 Data')

    # Data args
    parser.add_argument('--subject_id', type=int, default=1, help='Subject ID to train/validate on')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATASET_DIR, help='Directory containing PAMAP2 .dat files')
    parser.add_argument('--sensor_prefix', type=str, default=None, help='Optional prefix to filter sensor columns (e.g., "acc", "gyro")')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length of sliding windows (T)')
    parser.add_argument('--window_step', type=int, default=50, help='Step size for sliding window')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')

    # Model args
    parser.add_argument('--d_model', type=int, default=128, help='Internal dimension of the model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed-forward layer')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Total number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of layers to replace with MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # num_classes is determined from data

    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Hidden dimension within experts')

    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory to save results/models')

    # Loss args
    parser.add_argument('--lambda_load', type=float, default=0.1, help='Weight for load balancing loss')

    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='pamap-baseline-moe', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (defaults to baseline_subj{subject_id}_seq{seq_len}_moe{num_moe_layers})')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb even if --use_wandb is set (useful for debugging)')
    parser.add_argument('--wandb_log_model', action='store_true', help='Log best model as wandb artifact')
    parser.add_argument('--wandb_sweep', action='store_true', help='Enable hyperparameter sweep mode')

    # Distributed training args
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--cuda_device', type=int, default=0, help='Specify which GPU to use when running in single-GPU mode')

    args = parser.parse_args()
    
    # If wandb sweep is enabled, get hyperparameters from wandb
    if args.use_wandb and args.wandb_sweep:
        # Initialize wandb in sweep mode
        wandb.init()
        # Update args with sweep config
        sweep_config = wandb.config
        for key, value in sweep_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    main(args)
