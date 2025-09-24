import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
from typing import Dict, List
import argparse
import pickle
import numpy as np
import random
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb
from datetime import datetime
from plot_expert_activation import analyze_expert_activations
from wesad_rus_multimodal import load_wesad_data
from trus_moe_multimodal import MultimodalTRUSMoEModel
from trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss

def load_wesad_rus_data(rus_filepath: str, modality_names: List[str], seq_len: int) -> Dict[str, torch.Tensor]:
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")
    
    print(f"Loading RUS data from: {rus_filepath}")
    all_pid_results = np.load(rus_filepath, allow_pickle=True)
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}

    T = seq_len

    U = torch.zeros(num_modalities, T, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)

    processed_pairs = set()

    for result in all_pid_results:
        mod1, mod2 = result['feature_pair']
        
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because one or both modalities not found.")
            continue
        
        m1_idx = modality_to_idx[mod1]
        m2_idx = modality_to_idx[mod2]
        
        if m1_idx == m2_idx:
            continue
        
        pair_key = (min(m1_idx, m2_idx), max(m1_idx, m2_idx))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        lag_results = result['lag_results']
        num_lags = len(lag_results)
        segment_length = max(1, T // num_lags)
        
        for lag_idx, lag_data in enumerate(lag_results):
            start_idx = lag_idx * segment_length
            end_idx = min(T, (lag_idx + 1) * segment_length)
            
            if lag_idx == num_lags - 1:
                end_idx = T
                
            if start_idx >= T:
                break
                
            R_value = lag_data['R_value']
            S_value = lag_data['S_value']
            U1_value = lag_data['U1_value']
            U2_value = lag_data['U2_value']
            
            R[m1_idx, m2_idx, start_idx:end_idx] = torch.tensor(R_value)
            R[m2_idx, m1_idx, start_idx:end_idx] = torch.tensor(R_value)

            S[m1_idx, m2_idx, start_idx:end_idx] = torch.tensor(S_value)
            S[m2_idx, m1_idx, start_idx:end_idx] = torch.tensor(S_value)

            current_segment_U1 = U[m1_idx, start_idx:end_idx]
            current_max_U1 = current_segment_U1.max().item()
            new_value_U1 = max(current_max_U1, U1_value)
            U[m1_idx, start_idx:end_idx] = torch.tensor(new_value_U1)

            current_segment_U2 = U[m2_idx, start_idx:end_idx]
            current_max_U2 = current_segment_U2.max().item()
            new_value_U2 = max(current_max_U2, U2_value)
            U[m2_idx, start_idx:end_idx] = torch.tensor(new_value_U2)
    print(f"Modality-level RUS data computed. Shapes: U({U.shape}), R({R.shape}), S({S.shape})")
    print(f"  Average R value: {R.mean().item():.4f}")
    print(f"  Average S value: {S.mean().item():.4f}")
    print(f"  Average U value: {U.mean().item():.4f}")

    return {'U': U, 'R': R, 'S': S}


class MultimodalWESADDataset(Dataset):
    def __init__(self, subjects, processed_dataset_dir, rus_data, modality_names, seq_len, step):
        
        self.subjects = subjects
        self.processed_dataset_dir = processed_dataset_dir
        self.subjects_data = load_wesad_data(subjects, processed_dataset_dir)
        self.rus_data = rus_data
        self.modality_names = modality_names
        self.seq_len = seq_len
        self.step = step
        
        self.windows = []
        self.labels = []

        self._create_windows()



    def _create_windows_for_subject(self, subject_data):
        
        modality_pairs = list(itertools.combinations(self.modality_names, 2))
        for mod1, mod2 in modality_pairs:
            assert len(subject_data[mod1]) == len(subject_data[mod2])
            assert len(subject_data[mod1]) == len(subject_data['labels'])

        if len(subject_data[self.modality_names[0]]) < self.seq_len:
            return
        
        
        
        label_values = subject_data['labels']
        total_samples = len(subject_data[self.modality_names[0]])
        for i in range(0, total_samples - self.seq_len + 1, self.step):
            modality_signals = []
            for mod_name in self.modality_names:
                modality_signals.append(torch.tensor(subject_data[mod_name][i : i + self.seq_len], dtype=torch.float32))
        
            window_labels = label_values[i : i + self.seq_len]
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]
            
            self.windows.append(modality_signals)
            self.labels.append(torch.tensor(most_frequent_label, dtype=torch.long))

    def _create_windows(self):
        for subject_data in tqdm(self.subjects_data, desc="Creating windows"):
            self._create_windows_for_subject(subject_data)

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.rus_data, self.labels[idx]

def collate_multimodal(batch):
    """
    Custom collate function for multimodal data.
    """
    modality_data_lists = [[] for _ in range(len(batch[0][0]))]  # One list per modality
    rus_data_batch = {'U': [], 'R': [], 'S': []}
    labels = []
    
    for item in batch:
        modality_tensors, rus_dict, label = item
        
        # Collect modality data
        for i, mod_tensor in enumerate(modality_tensors):
            modality_data_lists[i].append(mod_tensor)
        
        # Collect RUS data
        rus_data_batch['U'].append(rus_dict['U'])
        rus_data_batch['R'].append(rus_dict['R'])
        rus_data_batch['S'].append(rus_dict['S'])
        
        # Collect labels
        labels.append(label)
    
    # Stack data
    modality_batches = [torch.stack(mod_list) for mod_list in modality_data_lists]
    rus_batches = {k: torch.stack(v) for k, v in rus_data_batch.items()}
    label_batch = torch.stack(labels)
    
    return modality_batches, rus_batches, label_batch

def train_epoch_multimodal_wesad(model: MultimodalTRUSMoEModel,
                                 dataloader: DataLoader,
                                 optimizer: optim.Optimizer,
                                 task_criterion: nn.Module,
                                 device: torch.device,
                                 args: argparse.Namespace,
                                 current_epoch: int):
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

    for batch_idx, (modality_data, rus_values_batch, labels) in enumerate(progress_bar):
        modality_data = [mod.to(device) for mod in modality_data]
        rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
        labels = labels.to(device)

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

        total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load

        total_loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        optimizer.step()

        total_loss_accum += total_loss.item()
        task_loss_accum += task_loss.item()
        unique_loss_accum += total_L_unique.item()
        redundancy_loss_accum += total_L_redundancy.item()
        synergy_loss_accum += total_L_synergy.item()
        load_loss_accum += total_L_load.item()

        predictions = torch.argmax(final_logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        if total_samples > 0:
            current_acc = 100. * correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'TaskL': f"{task_loss.item():.4f}",
                'Acc': f"{current_acc:.2f}%"
            })
    
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
    
    return avg_total_loss, accuracy

def eval_epoch_multimodal_wesad(model: MultimodalTRUSMoEModel,
                                    dataloader: DataLoader,
                                    task_criterion: nn.Module,
                                    device: torch.device,
                                    args: argparse.Namespace,
                                    current_epoch: int):
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Val]", leave=False)
    
    with torch.no_grad():
        for batch_idx, (modality_data, rus_values_batch, labels) in enumerate(progress_bar):
            # Move data to device
            modality_data = [mod.to(device) for mod in modality_data]
            rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
            labels = labels.to(device)
            
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
        args.run_name = f"wesad_multimodal_rus_moe_{timestamp}"

    # Set up wandb

    if args.use_wandb:        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=args.run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )


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


    class_idx_to_name = {0: 'Transient',
                         1: 'Baseline',
                         2: 'Stress',
                         3: 'Amusement',
                         4: 'Meditation',
                         5: 'Unknown'}
    num_classes = len(class_idx_to_name)
    
    data_split = pickle.load(open(os.path.join(args.processed_dataset_dir, 'wesad_split.pkl'), 'rb'))
    train_subjects = data_split['train']
    val_subjects = data_split['val']

    modality_dim_dict = {'chest_signals': 8, 'wrist_signals': 6}
    modality_names = list(modality_dim_dict.keys())
    rus_data = load_wesad_rus_data(args.rus_data_path, modality_names, args.seq_len)

    train_dataset = MultimodalWESADDataset(train_subjects, args.processed_dataset_dir, rus_data, modality_names, args.seq_len, args.window_step)
    val_dataset = MultimodalWESADDataset(val_subjects, args.processed_dataset_dir, rus_data, modality_names, args.seq_len, args.window_step)
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    )

    # Calculate no-skill baseline accuracy
    print("Calculating no-skill baseline accuracy...")
    
    # Get all training labels
    train_labels = []
    for _, _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)
    
    # Get all validation labels
    val_labels = []
    for _, _, labels in val_loader:
        val_labels.extend(labels.numpy())
    val_labels = np.array(val_labels)
    
    # Calculate most frequent class in training set
    train_unique_labels, train_counts = np.unique(train_labels, return_counts=True)
    most_frequent_class = train_unique_labels[np.argmax(train_counts)]
    most_frequent_count = np.max(train_counts)
    
    # Calculate baseline accuracies
    train_baseline_acc = (most_frequent_count / len(train_labels)) * 100
    val_baseline_acc = (np.sum(val_labels == most_frequent_class) / len(val_labels)) * 100
    
    print(f"Training set: {len(train_labels)} samples")
    print(f"  Most frequent class: {class_idx_to_name[most_frequent_class]} (class {most_frequent_class})")
    print(f"  Most frequent class count: {most_frequent_count}")
    print(f"  No-skill baseline accuracy: {train_baseline_acc:.2f}%")
    
    print(f"Validation set: {len(val_labels)} samples")
    print(f"  No-skill baseline accuracy: {val_baseline_acc:.2f}%")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "baseline/train_accuracy": train_baseline_acc,
            "baseline/val_accuracy": val_baseline_acc,
            "baseline/most_frequent_class": most_frequent_class,
            "baseline/train_samples": len(train_labels),
            "baseline/val_samples": len(val_labels)
        })
    
    print("-" * 50)

    

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

    model = MultimodalTRUSMoEModel(
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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Learning rate scheduler
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    os.makedirs(os.path.join(args.output_dir, 'checkpoints', args.run_name), exist_ok=True)
    task_criterion = nn.CrossEntropyLoss()
    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_multimodal_wesad(
            model, train_loader, optimizer, task_criterion, device, args, epoch
        )
        
        val_loss, val_acc = eval_epoch_multimodal_wesad(
            model, val_loader, task_criterion, device, args, epoch
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            
            save_path = os.path.join(args.output_dir, 'checkpoints', args.run_name, f'best_multimodal_model_wesad.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'args': args,
                'modality_configs': modality_configs,
                'modality_names': modality_names,
            }, save_path)
            print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved.")
            
            if args.use_wandb:
                wandb.log({"best_val_accuracy": best_val_accuracy, "best_epoch": epoch + 1})
        
        if scheduler is not None:
            scheduler.step()
    
    print("Training finished.")

    if best_epoch != -1:
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")
        
        if args.plot_expert_activations and len(val_loader) > 0:
            print("\nGenerating expert activation plots for the best multimodal TRUS-MoE model...")
            
            best_model_path = os.path.join(args.output_dir, 'checkpoints', args.run_name, f'best_multimodal_model_wesad.pth')

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
                    num_classes=num_classes,
                    max_seq_len=args.seq_len,
                    dropout=args.dropout,
                    use_checkpoint=args.use_gradient_checkpointing,
                    output_attention=False
                ).to(device)
                plot_model.load_state_dict(checkpoint['model_state_dict'])
                plot_model.eval()

                print(f"Getting {args.plot_num_samples} samples from validation dataloader for expert activation plotting...")
                
                plot_loader = DataLoader(
                    val_dataset,
                    batch_size=args.plot_num_samples,
                    shuffle=True,
                    num_workers=0,
                    collate_fn=collate_multimodal
                )
                
                plot_iter = iter(plot_loader)
                batch_modalities, batch_rus, batch_labels = next(plot_iter)
                # Move to device (same as training/validation loops)
                batch_modalities = [mod.to(device) for mod in batch_modalities]
                batch_rus = {k: v.to(device) for k, v in batch_rus.items()}
                
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

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multimodal TRUS-MoE Model on WESAD Data')
    parser.add_argument('--output_dir', type=str, default='../results/wesad',
                        help='Directory to save analysis results')
    parser.add_argument('--processed_dataset_dir', type=str, required=True,
                        help='Root directory of the processed WESAD datasets')
    parser.add_argument('--rus_data_path', type=str, required=True,
                        help='Path to the RUS data file')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Length of sequences for lag computation. If None, inferred from data')
    parser.add_argument('--window_step', type=int, default=25, help='Step size for sliding window')


    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID to use. If None, will use CPUs.')

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
    parser.add_argument('--moe_num_synergy_experts', type=int, default=2, help='Number of synergy experts')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Expert hidden dimension')
    parser.add_argument('--moe_capacity_factor', type=float, default=1.25, help='Expert capacity factor')
    parser.add_argument('--moe_drop_tokens', action='store_true', help='Drop tokens exceeding capacity')
    
    # MoE router args
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=64, help='GRU hidden dim in router')
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=64,
                        help='Token processing dim in router')
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=32, help='Attention key dim in router')
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=32,
                        help='Attention value dim in router')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use cosine annealing LR scheduler')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    
    # Loss args
    parser.add_argument('--threshold_u', type=float, default=0.5, help='Threshold for uniqueness loss')
    parser.add_argument('--threshold_r', type=float, default=0.1, help='Threshold for redundancy loss')
    parser.add_argument('--threshold_s', type=float, default=0.1, help='Threshold for synergy loss')
    parser.add_argument('--lambda_u', type=float, default=1, help='Weight for uniqueness loss')
    parser.add_argument('--lambda_r', type=float, default=1, help='Weight for redundancy loss')
    parser.add_argument('--lambda_s', type=float, default=1, help='Weight for synergy loss')
    parser.add_argument('--lambda_load', type=float, default=0.02, help='Weight for load balancing loss')
    parser.add_argument('--epsilon_loss', type=float, default=1e-8, help='Epsilon for loss stability')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='wesad-multimodal-trus-moe', 
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--run_name', type=str, default=None, help='run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')

    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true', help='Generate expert activation plots after training')
    parser.add_argument('--plot_num_samples', type=int, default=32, help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    main(args)