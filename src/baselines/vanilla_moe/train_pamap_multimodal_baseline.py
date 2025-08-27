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
    from baseline_moe_multimodal import MultimodalBaselineMoEModel, compute_total_loss
    from pamap_rus import get_pamap_column_names, load_pamap_data, preprocess_pamap_data
    from plot_expert_activation import analyze_expert_activations
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Please ensure baseline_moe_multimodal.py, pamap_rus.py, and plot_expert_activation.py are accessible.")
    sys.exit(1)


DEFAULT_DATASET_DIR = "/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol"
DEFAULT_OUTPUT_DIR = "../results/pamap_multimodal_baseline_training"


def categorize_pamap_sensors(sensor_columns: List[str]) -> Dict[str, List[str]]:
    """
    Categorizes PAMAP sensor columns by body location.
    
    Expected sensor naming convention:
    - {sensor_type}_{location}_{axis}
    - e.g., acc_chest_x, gyro_hand_y, mag_ankle_z
    
    Returns:
        Dictionary mapping modality names to lists of sensor columns
    """
    modality_sensors = {
        'chest': [],
        'hand': [],
        'ankle': [],
        'heart_rate': []
    }
    
    for col in sensor_columns:
        col_lower = col.lower()
        
        # Check for heart rate
        if 'heart' in col_lower or 'hr' in col_lower:
            modality_sensors['heart_rate'].append(col)
        # Check for body locations
        elif 'chest' in col_lower:
            modality_sensors['chest'].append(col)
        elif 'hand' in col_lower:
            modality_sensors['hand'].append(col)
        elif 'ankle' in col_lower:
            modality_sensors['ankle'].append(col)
        else:
            # Try to infer from other patterns
            # Sometimes sensors might be named differently
            print(f"Warning: Could not categorize sensor column: {col}")
    
    # Remove empty modalities
    modality_sensors = {k: v for k, v in modality_sensors.items() if v}
    
    return modality_sensors


class MultimodalPamapBaselineDataset(Dataset):
    """
    PyTorch Dataset for multimodal PAMAP2 activity recognition.
    Provides separate features for each modality without RUS values.
    """
    def __init__(self, subject_id: int, data_dir: str,
                 modality_sensors: Dict[str, List[str]], seq_len: int, step: int,
                 activity_map: Dict[int, int]):
        """
        Args:
            subject_id: The subject ID to load data for.
            data_dir: Directory containing PAMAP2 .dat files.
            modality_sensors: Dictionary mapping modality names to sensor columns.
            seq_len: Length of the sliding window (T).
            step: Step size for the sliding window.
            activity_map: Dictionary mapping original activity IDs to 0-based indices.
        """
        self.subject_id = subject_id
        self.data_dir = data_dir
        self.modality_sensors = modality_sensors
        self.modality_names = list(modality_sensors.keys())
        self.num_modalities = len(self.modality_names)
        self.seq_len = seq_len
        self.step = step
        self.activity_map = activity_map
        
        # Get all sensor columns in order
        all_sensor_columns = []
        for sensors in modality_sensors.values():
            all_sensor_columns.extend(sensors)
        
        try:
            df = load_pamap_data(self.subject_id, self.data_dir)
            # Use only selected sensor columns + activity_id
            selected_cols_with_id = ['timestamp', 'activity_id'] + all_sensor_columns
            cols_to_use = [col for col in selected_cols_with_id if col in df.columns]
            missing_cols = set(selected_cols_with_id) - set(cols_to_use)
            
            if missing_cols:
                print(f"Warning: Requested sensor columns not found in data: {missing_cols}")
            
            df_subset = df[cols_to_use].copy()
            self.df_processed, _ = preprocess_pamap_data(df_subset)
            
            # Remap activity IDs
            self.df_processed['activity_label'] = self.df_processed['activity_id'].map(activity_map)
            self.df_processed.dropna(subset=['activity_label'], inplace=True)
            self.df_processed['activity_label'] = self.df_processed['activity_label'].astype(int)
            
        except Exception as e:
            print(f"Error processing data for subject {subject_id}: {e}")
            raise
        
        if self.df_processed.empty:
            print(f"Warning: No data remaining for subject {subject_id}")
            self.windows = []
            self.labels = []
        else:
            self._create_windows()
    
    def _create_windows(self):
        """Creates sliding windows with separate features for each modality."""
        self.windows = []
        self.labels = []
        
        # Get data for each modality
        modality_data = {}
        for mod_name, sensors in self.modality_sensors.items():
            # Get only the sensors that exist in the dataframe
            existing_sensors = [s for s in sensors if s in self.df_processed.columns]
            if existing_sensors:
                modality_data[mod_name] = self.df_processed[existing_sensors].values
            else:
                print(f"Warning: No sensors found for modality {mod_name}")
                # Create dummy data with zeros
                modality_data[mod_name] = np.zeros((len(self.df_processed), 1))
        
        label_values = self.df_processed['activity_label'].values
        total_samples = len(self.df_processed)
        
        for i in range(0, total_samples - self.seq_len + 1, self.step):
            window_data_by_modality = []
            
            for mod_name in self.modality_names:
                mod_window = modality_data[mod_name][i : i + self.seq_len]  # (T, D_m)
                window_data_by_modality.append(torch.tensor(mod_window, dtype=torch.float32))
            
            window_labels = label_values[i : i + self.seq_len]
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]
            
            self.windows.append(window_data_by_modality)
            self.labels.append(torch.tensor(most_frequent_label, dtype=torch.long))
        
        print(f"Created {len(self.windows)} multimodal windows for subject {self.subject_id}.")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # Returns: list of modality tensors, label (no RUS values)
        return self.windows[idx], self.labels[idx]


def collate_multimodal_baseline(batch):
    """
    Custom collate function for multimodal baseline data.
    """
    modality_data_lists = [[] for _ in range(len(batch[0][0]))]  # One list per modality
    labels = []
    
    for item in batch:
        modality_tensors, label = item
        
        # Collect modality data
        for i, mod_tensor in enumerate(modality_tensors):
            modality_data_lists[i].append(mod_tensor)
        
        # Collect labels
        labels.append(label)
    
    # Stack data
    modality_batches = [torch.stack(mod_list) for mod_list in modality_data_lists]
    label_batch = torch.stack(labels)
    
    return modality_batches, label_batch


def train_epoch_multimodal_baseline(model: MultimodalBaselineMoEModel,
                                  dataloader: DataLoader,
                                  optimizer: optim.Optimizer,
                                  task_criterion: nn.Module,
                                  device: torch.device,
                                  args: argparse.Namespace,
                                  current_epoch: int):
    """Runs one training epoch for multimodal baseline model."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    load_balance_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Set epoch for distributed sampler if using
    if args.distributed and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(current_epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Train]", 
                        leave=False, disable=args.distributed and dist.get_rank() != 0)
    
    for batch_idx, (modality_data, labels) in enumerate(progress_bar):
        # Move data to device
        modality_data = [mod.to(device) for mod in modality_data]
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        final_logits, all_aux_moe_outputs = model(modality_data)
        
        # Calculate losses using baseline model's loss function
        loss_dict = compute_total_loss(final_logits, labels, all_aux_moe_outputs, 
                                     load_balance_weight=args.load_balance_weight)
        
        total_loss = loss_dict['total_loss']
        task_loss = loss_dict['classification_loss']
        load_balance_loss = loss_dict['load_balance_loss']
        
        # Backward pass and optimize
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN or Inf detected in total loss at batch {batch_idx}. Skipping.")
        else:
            total_loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            
            # Accumulate losses and accuracy
            total_loss_accum += total_loss.item()
            task_loss_accum += task_loss.item()
            load_balance_loss_accum += load_balance_loss.item()
            
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            if total_samples > 0:
                current_acc = 100. * correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'TaskL': f"{task_loss.item():.4f}",
                    'LoadBalL': f"{load_balance_loss.item():.4f}",
                    'Acc': f"{current_acc:.2f}%"
                })
    
    # Calculate average losses and accuracy
    num_batches = len(dataloader)
    if num_batches == 0:
        return 0.0, 0.0
    
    avg_total_loss = total_loss_accum / num_batches
    avg_task_loss = task_loss_accum / num_batches
    avg_load_balance_loss = load_balance_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Log metrics to wandb
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb.log({
            "train/total_loss": avg_total_loss,
            "train/task_loss": avg_task_loss,
            "train/load_balance_loss": avg_load_balance_loss,
            "train/accuracy": accuracy,
            "epoch": current_epoch + 1
        })
    
    print(f"Epoch {current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, "
          f"Task Loss: {avg_task_loss:.4f}, Load Balance Loss: {avg_load_balance_loss:.4f}, "
          f"Accuracy: {accuracy:.2f}%")
    
    return avg_total_loss, accuracy


def validate_epoch_multimodal_baseline(model: MultimodalBaselineMoEModel,
                                     dataloader: DataLoader,
                                     task_criterion: nn.Module,
                                     device: torch.device,
                                     args: argparse.Namespace,
                                     current_epoch: int):
    """Runs one validation epoch for multimodal baseline model."""
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Val]", leave=False)
    
    with torch.no_grad():
        for batch_idx, (modality_data, labels) in enumerate(progress_bar):
            # Move data to device
            modality_data = [mod.to(device) for mod in modality_data]
            labels = labels.to(device)
            
            # Forward pass
            final_logits, _ = model(modality_data)
            
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
    """Main function to set up and run multimodal baseline training."""
    # Initialize wandb if enabled
    if args.use_wandb and (not args.distributed or (args.distributed and dist.get_rank() == 0)):
        wandb_config = {k: v for k, v in vars(args).items()}
        run_name = f"baseline_subj{args.subject_id}_seq{args.seq_len}"
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
            raise ValueError("For distributed training, please launch with torchrun")
        
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ.get('RANK', local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl')
        is_main_process = global_rank == 0
        
        if is_main_process:
            print(f"Distributed training initialized with world_size: {world_size}")
    else:
        # Non-distributed mode
        if args.cuda_device >= 0 and torch.cuda.is_available() and not args.no_cuda:
            device = torch.device(f"cuda:{args.cuda_device}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        is_main_process = True
    
    if is_main_process:
        print(f"Using device: {device}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load initial data to determine sensors and activities
    print("Loading initial data to determine sensors and activities...")
    try:
        temp_df = load_pamap_data(args.subject_id, args.data_dir)
        temp_df_processed, all_sensor_columns = preprocess_pamap_data(temp_df)
        
        # Categorize sensors by modality
        modality_sensors = categorize_pamap_sensors(all_sensor_columns)
        
        if not modality_sensors:
            print("Error: No sensors could be categorized into modalities.")
            sys.exit(1)
        
        print(f"Found {len(modality_sensors)} modalities:")
        for mod_name, sensors in modality_sensors.items():
            print(f"  {mod_name}: {len(sensors)} sensors")
        
        # Determine unique activities and create mapping
        unique_activities = sorted([act for act in temp_df_processed['activity_id'].unique() if act != 0])
        activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}
        num_classes = len(activity_map)
        print(f"Found {num_classes} activities: {unique_activities}")
        
    except Exception as e:
        print(f"Error during initial data loading: {e}")
        sys.exit(1)
    
    # Create dataset (no RUS data needed for baseline)
    if is_main_process:
        print("Creating multimodal baseline dataset...")
    
    try:
        full_dataset = MultimodalPamapBaselineDataset(
            subject_id=args.subject_id,
            data_dir=args.data_dir,
            modality_sensors=modality_sensors,
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
            print("Error: Dataset is empty after processing.")
        if args.distributed:
            dist.destroy_process_group()
        sys.exit(1)
    
    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    if is_main_process:
        print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")
    
    # Create samplers for distributed training
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                          rank=dist.get_rank(), shuffle=True, seed=args.seed)
        val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                        rank=dist.get_rank(), shuffle=False, seed=args.seed) if val_size > 0 else None
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal_baseline
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal_baseline
    ) if val_size > 0 else []
    
    # Model configuration
    modality_names = list(modality_sensors.keys())
    modality_configs = []
    
    for mod_name in modality_names:
        num_sensors = len(modality_sensors[mod_name])
        config = {
            'input_dim': num_sensors,
            'num_layers': args.modality_encoder_layers,
            'nhead': args.nhead,
            'd_ff': args.d_ff,
            'use_cnn': args.use_cnn_encoders and mod_name != 'heart_rate',  # No CNN for heart rate
            'kernel_size': 3
        }
        modality_configs.append(config)
    
    # MoE configuration for baseline model
    moe_layer_config = {
        "num_experts": args.moe_num_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "router_noise_std": args.moe_router_noise_std,
        "load_balance_weight": args.load_balance_weight,
        "use_load_balancing": args.use_load_balancing,
        "capacity_factor": args.moe_capacity_factor,
        "drop_tokens": args.moe_drop_tokens,
    }
    
    if is_main_process:
        print("Initializing multimodal baseline MoE model...")
    
    model = MultimodalBaselineMoEModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_classes=num_classes,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_checkpoint=args.use_gradient_checkpointing,
        output_attention=False
    ).to(device)
    
    # Wrap model with DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                   find_unused_parameters=False)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    task_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if is_main_process:
        print("Starting training...")
    
    best_val_accuracy = -1.0
    best_epoch = -1
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_multimodal_baseline(
            model, train_loader, optimizer, task_criterion, device, args, epoch
        )
        
        if len(val_loader) > 0:
            val_loss, val_acc = validate_epoch_multimodal_baseline(
                model, val_loader, task_criterion, device, args, epoch
            )
            
            # For distributed training, all processes should have same val_acc
            if args.distributed:
                val_acc_tensor = torch.tensor([val_acc], device=device)
                dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
                val_acc = val_acc_tensor.item() / dist.get_world_size()
            
            # Save best model
            if val_acc > best_val_accuracy and is_main_process:
                best_val_accuracy = val_acc
                best_epoch = epoch
                
                save_path = os.path.join(args.output_dir, f'best_baseline_model_subj{args.subject_id}.pth')
                
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
                    'args': args,
                    'modality_configs': modality_configs,
                    'modality_names': modality_names,
                }, save_path)
                
                print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved.")
                
                # Log best model to wandb
                if args.use_wandb:
                    wandb.log({"best_val_accuracy": best_val_accuracy, "best_epoch": epoch + 1})
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
    
    # Synchronize processes before finishing
    if args.distributed:
        dist.barrier()
    
    if is_main_process:
        print("Training finished.")
        if best_epoch != -1:
            print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")
            
            # Generate expert activation plots for the best model
            if args.plot_expert_activations and val_size > 0:
                print("\nGenerating expert activation plots for the best multimodal baseline MoE model...")
                
                # Load the best model
                best_model_path = os.path.join(args.output_dir, f'best_baseline_model_subj{args.subject_id}.pth')
                if os.path.exists(best_model_path):
                    # Add argparse.Namespace to safe globals for PyTorch 2.6+ compatibility
                    torch.serialization.add_safe_globals([argparse.Namespace])
                    checkpoint = torch.load(best_model_path, map_location=device)
                    
                    # Create a fresh model instance
                    plot_model = MultimodalBaselineMoEModel(
                        modality_configs=modality_configs,
                        d_model=args.d_model,
                        nhead=args.nhead,
                        d_ff=args.d_ff,
                        num_encoder_layers=args.num_encoder_layers,
                        num_moe_layers=args.num_moe_layers,
                        moe_config=moe_layer_config,
                        num_classes=num_classes,
                        max_seq_len=args.seq_len,
                        dropout=args.dropout,
                        use_checkpoint=args.use_gradient_checkpointing,
                        output_attention=False
                    ).to(device)
                    
                    plot_model.load_state_dict(checkpoint['model_state_dict'])
                    plot_model.eval()
                    
                    # Get a batch of validation data
                    val_batch_modalities = []
                    num_plot_samples = min(args.plot_num_samples, len(val_dataset))
                    
                    for i in range(num_plot_samples):
                        modality_data, _ = val_dataset[i]
                        val_batch_modalities.append(modality_data)
                    
                    # Stack into batch format
                    batch_modalities = [[] for _ in range(len(val_batch_modalities[0]))]
                    for sample_modalities in val_batch_modalities:
                        for mod_idx, mod_data in enumerate(sample_modalities):
                            batch_modalities[mod_idx].append(mod_data)
                    
                    # Stack each modality
                    batch_modalities = [torch.stack(mod_list).to(device) for mod_list in batch_modalities]
                    
                    # Generate plots
                    plot_save_dir = os.path.join(args.output_dir, 'expert_activation_plots')
                    
                    # try:
                    analyze_expert_activations(
                        trus_model=None,
                        baseline_model=plot_model,
                        data_batch=batch_modalities,
                        rus_values=None,
                        modality_names=modality_names,
                        save_dir=plot_save_dir
                    )
                    print(f"Expert activation plots saved to {plot_save_dir}")
                    # except Exception as e:
                    #     print(f"Error generating expert activation plots: {e}")
                        
                else:
                    print(f"Best model checkpoint not found at {best_model_path}")
            
        else:
            print("Training finished (no validation performed or no improvement).")
    
    # Finish wandb run
    if args.use_wandb and is_main_process:
        wandb.finish()
    
    # Clean up distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multimodal Baseline MoE Model on PAMAP2 Data')
    
    # Data args
    parser.add_argument('--subject_id', type=int, default=1, help='Subject ID to train/validate on')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATASET_DIR, 
                       help='Directory containing PAMAP2 .dat files')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length (T)')
    parser.add_argument('--window_step', type=int, default=50, help='Step size for sliding window')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    
    # Model architecture args
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--modality_encoder_layers', type=int, default=2, 
                       help='Number of layers in modality-specific encoders')
    parser.add_argument('--use_cnn_encoders', action='store_true', 
                       help='Use CNN layers in modality encoders')
        
    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Expert hidden dimension')
    parser.add_argument('--moe_capacity_factor', type=float, default=1.25, help='Expert capacity factor')
    parser.add_argument('--moe_drop_tokens', action='store_true', help='Drop tokens exceeding capacity')
    parser.add_argument('--moe_router_noise_std', type=float, default=0.1, help='Router noise standard deviation')
    parser.add_argument('--use_load_balancing', action='store_true', default=True, help='Use load balancing loss')
    parser.add_argument('--load_balance_weight', type=float, default=0.01, help='Load balancing loss weight')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, 
                       help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use cosine annealing LR scheduler')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', 
                       help='Use gradient checkpointing to save memory')
    
    # System args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--cuda_device', type=int, default=0, help='Specific GPU to use')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                       help='Directory to save results/models')
    
    # Distributed training args
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='pamap-multimodal-baseline-moe', 
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')
    
    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true', help='Generate expert activation plots after training')
    parser.add_argument('--plot_num_samples', type=int, default=32, help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    main(args)