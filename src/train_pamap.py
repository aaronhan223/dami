import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import math
import os
import sys
import argparse
import random
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Set

# --- Imports from project files ---
# Assume these files are in the same directory or PYTHONPATH is set correctly
try:
    from trus_moe_model import TRUSMoEModel_LargeScale, TRUSMoEBlock, calculate_rus_losses, calculate_load_balancing_loss, JSD
    from pamap_rus import get_pamap_column_names, load_pamap_data, preprocess_pamap_data
except ImportError as e:
    print(f"Error importing project files: {e}")
    print("Please ensure trus_moe_model.py and pamap_rus.py are accessible.")
    sys.exit(1)

# --- Constants ---
# Default values, can be overridden by args
DEFAULT_DATASET_DIR = "/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol"
DEFAULT_RUS_FILE_PATTERN = "../results/pamap/pamap_subject{SUBJECT_ID}_lag{MAX_LAG}_bins{BINS}_thresh{DOMINANCE_THRESHOLD:.1f}.npy"
DEFAULT_OUTPUT_DIR = "../results/pamap_training"


# --- Helper Functions (Adapted from trus_moe_model.py) ---
# Note: calculate_rus_losses, calculate_load_balancing_loss, JSD are imported above


# --- Data Loading and Processing ---

def load_simplified_rus_data(rus_filepath: str, sensor_columns: List[str], seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Loads the pre-computed dominant RUS terms from the .npy file and creates
    simplified, dense tensors (U, R, S) usable by the model's auxiliary losses.

    Strategy: Use binary indicators based on the *first* dominant term found for a pair.
              Propagate this indicator across the entire sequence length T.
              U is marked per modality if *any* pair involving it shows dominance.

    Args:
        rus_filepath: Path to the .npy file containing dominant PID results.
        sensor_columns: List of sensor names (modalities).
        seq_len: The sequence length (T) for the tensors.

    Returns:
        A dictionary containing 'U', 'R', 'S' tensors.
        'U': Tensor shape (M, T)
        'R': Tensor shape (M, M, T)
        'S': Tensor shape (M, M, T)
        Where M is the number of modalities (sensors).
    """
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")

    print(f"Loading simplified RUS data from: {rus_filepath}")
    dominant_pid_results = np.load(rus_filepath, allow_pickle=True)

    M = len(sensor_columns)
    T = seq_len
    sensor_to_idx = {name: i for i, name in enumerate(sensor_columns)}

    # Initialize tensors with zeros
    U = torch.zeros(M, T, dtype=torch.float32)
    R = torch.zeros(M, M, T, dtype=torch.float32)
    S = torch.zeros(M, M, T, dtype=torch.float32)

    processed_pairs = set() # Track pairs to use only the first dominant term found

    for result in dominant_pid_results:
        col1, col2 = result['feature_pair']
        dominant_term = result['dominant_term']
        lag = result['lag'] # Currently unused in this simplified version

        if col1 not in sensor_to_idx or col2 not in sensor_to_idx:
            # print(f"Warning: Sensor pair ({col1}, {col2}) from RUS file not in selected sensor columns. Skipping.")
            continue

        m1 = sensor_to_idx[col1]
        m2 = sensor_to_idx[col2]
        pair_key = tuple(sorted((m1, m2)))

        # Only process the first dominant term encountered for a pair
        if pair_key not in processed_pairs:
            processed_pairs.add(pair_key)
            if dominant_term == 'R':
                R[m1, m2, :] = 1.0
                R[m2, m1, :] = 1.0
            elif dominant_term == 'S':
                S[m1, m2, :] = 1.0
                S[m2, m1, :] = 1.0
            elif dominant_term == 'U1': # Uniqueness associated with col1 (m1)
                U[m1, :] = 1.0
            elif dominant_term == 'U2': # Uniqueness associated with col2 (m2)
                U[m2, :] = 1.0

    print(f"Simplified RUS data loaded. Shapes: U({U.shape}), R({R.shape}), S({S.shape})")
    # Log some stats
    print(f"  Number of R=1 entries (per time step): {int(R.sum() / T / 2)}") # Divide by 2 for symmetry
    print(f"  Number of S=1 entries (per time step): {int(S.sum() / T / 2)}")
    print(f"  Number of U=1 entries (per time step): {int(U.sum() / T)}")

    return {'U': U, 'R': R, 'S': S}


class PamapWindowDataset(Dataset):
    """
    PyTorch Dataset for PAMAP2 activity recognition using sliding windows.
    Loads preprocessed data and aligns it with simplified RUS information.
    """
    def __init__(self, subject_id: int, data_dir: str, rus_data: Dict[str, torch.Tensor],
                 sensor_columns: List[str], seq_len: int, step: int,
                 activity_map: Dict[int, int]):
        """
        Args:
            subject_id: The subject ID to load data for.
            data_dir: Directory containing PAMAP2 .dat files.
            rus_data: Dictionary containing 'U', 'R', 'S' tensors (M, T or M, M, T).
            sensor_columns: List of sensor columns to use as modalities.
            seq_len: Length of the sliding window (T).
            step: Step size for the sliding window.
            activity_map: Dictionary mapping original activity IDs to 0-based indices.
        """
        self.subject_id = subject_id
        self.data_dir = data_dir
        self.rus_data = rus_data # Expected shapes: U(M,T), R(M,M,T), S(M,M,T)
        self.sensor_columns = sensor_columns
        self.seq_len = seq_len
        self.step = step
        self.activity_map = activity_map
        self.num_modalities = len(sensor_columns) # M
        self.input_dim = 1 # Each sensor value is a single feature

        try:
            df = load_pamap_data(self.subject_id, self.data_dir)
            # Use only selected sensor columns + activity_id
            selected_cols_with_id = ['activity_id'] + self.sensor_columns
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
            # If tied, takes the smallest label index. Add more sophisticated logic if needed.
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
        # Data shape (M, T, E_in=1), RUS dict (U:(M,T), R:(M,M,T), S:(M,M,T)), label (scalar)
        # Note: The RUS data is static for all windows in this simplified approach
        return self.windows[idx], self.rus_data, self.labels[idx]


# --- Training and Validation Loops ---

def train_epoch(model: TRUSMoEModel_LargeScale,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                task_criterion: nn.Module,
                device: torch.device,
                args: argparse.Namespace,
                current_epoch: int):
    """Runs one training epoch."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    unique_loss_accum = 0.0
    redundancy_loss_accum = 0.0
    synergy_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    num_moe_layers_in_model = sum(isinstance(layer, TRUSMoEBlock) for layer in model.layers)

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Train]", leave=False)
    for batch_idx, (data, rus_values_batch, labels) in enumerate(progress_bar):
        # data shape: (B, M, T, E_in)
        # rus_values_batch: Dict where each value is (B, M, T) or (B, M, M, T)
        # labels shape: (B,)
        data = data.to(device)
        # Move all tensors within the rus_values dict to the device
        # DataLoader default collate should stack the RUS tensors correctly along batch dim
        rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
        labels = labels.to(device)

        B, M, T, E_in = data.shape

        optimizer.zero_grad()

        # Forward pass
        final_logits, all_aux_moe_outputs = model(data, rus_values)

        # Calculate Task Loss
        task_loss = task_criterion(final_logits, labels)

        # Calculate Auxiliary Losses
        total_L_unique = torch.tensor(0.0, device=device)
        total_L_redundancy = torch.tensor(0.0, device=device)
        total_L_synergy = torch.tensor(0.0, device=device)
        total_L_load = torch.tensor(0.0, device=device)
        num_moe_layers_encountered = 0

        moe_aux_output_index = 0
        for layer in model.layers:
             if isinstance(layer, TRUSMoEBlock):
                if moe_aux_output_index < len(all_aux_moe_outputs):
                     aux_outputs = all_aux_moe_outputs[moe_aux_output_index]
                     moe_aux_output_index += 1
                else:
                    print(f"Warning: Mismatch in aux outputs at batch {batch_idx}")
                    continue

                num_moe_layers_encountered += 1
                gating_probs = aux_outputs['gating_probs'] # (B, M, T, N_exp)
                expert_indices = aux_outputs['expert_indices'] # (B, M, T, k)
                synergy_expert_indices = layer.moe_layer.synergy_expert_indices
                k = layer.moe_layer.k

                # RUS values already have batch dim from dataloader
                L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                    gating_probs, rus_values, synergy_expert_indices,
                    args.threshold_u, args.threshold_r, args.threshold_s,
                    args.lambda_u, args.lambda_r, args.lambda_s,
                    epsilon=args.epsilon_loss
                )
                L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, args.lambda_load)

                total_L_unique += L_unique
                total_L_redundancy += L_redundancy
                total_L_synergy += L_synergy
                total_L_load += L_load

        if num_moe_layers_encountered > 0:
            total_L_unique /= num_moe_layers_encountered
            total_L_redundancy /= num_moe_layers_encountered
            total_L_synergy /= num_moe_layers_encountered
            total_L_load /= num_moe_layers_encountered

        # Combine Losses
        total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load

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
            unique_loss_accum += total_L_unique.item() if torch.is_tensor(total_L_unique) else total_L_unique
            redundancy_loss_accum += total_L_redundancy.item() if torch.is_tensor(total_L_redundancy) else total_L_redundancy
            synergy_loss_accum += total_L_synergy.item() if torch.is_tensor(total_L_synergy) else total_L_synergy
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
    avg_unique_loss = unique_loss_accum / num_batches
    avg_redundancy_loss = redundancy_loss_accum / num_batches
    avg_synergy_loss = synergy_loss_accum / num_batches
    avg_load_loss = load_loss_accum / num_batches
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    print(f"Epoch {current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Aux Losses -> Unique: {avg_unique_loss:.4f}, Redundancy: {avg_redundancy_loss:.4f}, Synergy: {avg_synergy_loss:.4f}, Load: {avg_load_loss:.4f}")

    return avg_total_loss, accuracy


def validate_epoch(model: TRUSMoEModel_LargeScale,
                   dataloader: DataLoader,
                   task_criterion: nn.Module,
                   device: torch.device,
                   args: argparse.Namespace,
                   current_epoch: int):
    """Runs one validation epoch."""
    model.eval()
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{args.epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch_idx, (data, rus_values_batch, labels) in enumerate(progress_bar):
            data = data.to(device)
            rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
            labels = labels.to(device)

            # Forward pass
            final_logits, _ = model(data, rus_values) # Ignore aux outputs for validation

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

    print(f"Epoch {current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy

# --- Main Function ---

def main(args):
    """Main function to set up and run training."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Seed for reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(args.seed) # Use manual_seed_all for multi-GPU
        # Consider deterministic algorithms, might impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Output Directory ---
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


    # --- Load RUS Data ---
    rus_file = args.rus_file_pattern.format(
        SUBJECT_ID=args.subject_id,
        MAX_LAG=args.rus_max_lag,
        BINS=args.rus_bins,
        DOMINANCE_THRESHOLD=args.rus_dominance_threshold
    )
    try:
        rus_data_loaded = load_simplified_rus_data(rus_file, sensor_columns, args.seq_len)
    except FileNotFoundError as e:
        print(f"Error: {e}. Check RUS file path pattern and parameters.")
        sys.exit(1)
    except Exception as e:
         print(f"An error occurred loading RUS data: {e}")
         sys.exit(1)

    # --- Create Datasets and DataLoaders ---
    print("Creating dataset...")
    try:
        full_dataset = PamapWindowDataset(
            subject_id=args.subject_id,
            data_dir=args.data_dir,
            rus_data=rus_data_loaded,
            sensor_columns=sensor_columns,
            seq_len=args.seq_len,
            step=args.window_step,
            activity_map=activity_map
        )
    except Exception as e:
         print(f"Error creating dataset: {e}")
         sys.exit(1)

    if len(full_dataset) == 0:
        print("Error: Dataset is empty after processing and windowing.")
        sys.exit(1)

    # Split dataset into training and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    if train_size == 0 or val_size == 0:
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

    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False
    )
    # No shuffling for validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == torch.device("cuda") else False
    )
    print("Dataloaders created.")


    # --- Model Configuration ---
    num_modalities = len(sensor_columns) # M
    input_dim = 1 # E_in

    moe_router_config = {
        "gru_hidden_dim": args.moe_router_gru_hidden_dim,
        "token_processed_dim": args.moe_router_token_processed_dim,
        "attn_key_dim": args.moe_router_attn_key_dim,
        "attn_value_dim": args.moe_router_attn_value_dim,
    }
    moe_layer_config = {
        "input_dim": args.d_model,
        "output_dim": args.d_model,
        "num_experts": args.moe_num_experts,
        "num_synergy_experts": args.moe_num_synergy_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "synergy_expert_nhead": args.nhead,
        "router_config": moe_router_config,
    }

    print("Initializing model...")
    model = TRUSMoEModel_LargeScale(
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, task_criterion, device, args, epoch)

        if len(val_loader) > 0:
             val_loss, val_acc = validate_epoch(model, val_loader, task_criterion, device, args, epoch)

             # Simple best model saving based on validation accuracy
             if val_acc > best_val_accuracy:
                 best_val_accuracy = val_acc
                 best_epoch = epoch
                 save_path = os.path.join(args.output_dir, f'best_model_subj{args.subject_id}.pth')
                 torch.save({
                     'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'best_val_accuracy': best_val_accuracy,
                     'args': args # Save args for reference
                 }, save_path)
                 print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved to {save_path}")
        else:
             print(f"Epoch {epoch+1}: No validation set. Skipping validation and model saving.")


    print("Training finished.")
    if best_epoch != -1:
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")
    else:
        print("Training finished (no validation performed or no improvement).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRUS-MoE Model on PAMAP2 Data')

    # Data args
    parser.add_argument('--subject_id', type=int, default=1, help='Subject ID to train/validate on')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATASET_DIR, help='Directory containing PAMAP2 .dat files')
    parser.add_argument('--sensor_prefix', type=str, default=None, help='Optional prefix to filter sensor columns (e.g., "acc", "gyro")')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length of time-series windows (T)') # PAMAP freq is 100Hz, so 100 = 1 second
    parser.add_argument('--window_step', type=int, default=50, help='Step size for sliding window')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')

    # RUS Data Args
    parser.add_argument('--rus_file_pattern', type=str, default=DEFAULT_RUS_FILE_PATTERN, help='Pattern for locating the .npy RUS file')
    parser.add_argument('--rus_max_lag', type=int, default=10, help='Max lag used when generating RUS file (part of filename)')
    parser.add_argument('--rus_bins', type=int, default=8, help='Bins used when generating RUS file (part of filename)')
    parser.add_argument('--rus_dominance_threshold', type=float, default=0.3, help='Dominance threshold used when generating RUS file (part of filename)')

    # Model args (copied from trus_moe_model.py, adjust defaults if needed)
    parser.add_argument('--d_model', type=int, default=128, help='Internal dimension of the model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed-forward layer')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Total number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of layers to replace with MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # num_classes is determined from data

    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_num_synergy_experts', type=int, default=2, help='Number of experts designated as synergy experts')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Hidden dimension within experts')
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=64, help='GRU hidden dim in MoE router')
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=64, help='Token processing dim in MoE router')
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=32, help='Attention key dim in MoE router')
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=32, help='Attention value dim in MoE router')

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
    parser.add_argument('--threshold_u', type=float, default=0.7, help='Threshold for uniqueness loss (applied to simplified binary U)')
    parser.add_argument('--threshold_r', type=float, default=0.7, help='Threshold for redundancy loss (applied to simplified binary R)')
    parser.add_argument('--threshold_s', type=float, default=0.7, help='Threshold for synergy loss (applied to simplified binary S)')
    parser.add_argument('--lambda_u', type=float, default=0.01, help='Weight for uniqueness loss')
    parser.add_argument('--lambda_r', type=float, default=0.01, help='Weight for redundancy loss')
    parser.add_argument('--lambda_s', type=float, default=0.01, help='Weight for synergy loss')
    parser.add_argument('--lambda_load', type=float, default=0.01, help='Weight for load balancing loss')
    parser.add_argument('--epsilon_loss', type=float, default=1e-8, help='Epsilon for stability in loss calculations')

    args = parser.parse_args()
    main(args) 