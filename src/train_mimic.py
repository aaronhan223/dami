import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import os # For path joining
import pdb

# --- PyHealth Imports ---
# Make sure PyHealth is installed: pip install pyhealth
from pyhealth.datasets import MIMIC4Dataset # Use MIMIC-IV specific dataset
from pyhealth.tasks import mortality_prediction_mimic4_fn, length_of_stay_prediction_mimic4_fn, readmission_prediction_mimic4_fn # Example task function
from pyhealth.datasets.splitter import split_by_patient # Import split_by_patient
from pyhealth.datasets import get_dataloader # Import get_dataloader

# --- Model Imports ---
# Assume the model definitions from the previous artifact are in a file
# named 'trus_moe_model.py' in the same directory or accessible via PYTHONPATH
try:
    from trus_moe_model import (
        TRUSMoEModel_LargeScale,
        TRUSMoEBlock, # Needed for type checking in train/val loops
        calculate_rus_losses,
        calculate_load_balancing_loss
    )
    print("Successfully imported model components from trus_moe_model.py")
except ImportError as e:
    print(f"Error importing model components: {e}")
    print("Please ensure 'trus_moe_model.py' with model definitions is accessible.")
    # Define dummy classes/functions if import fails, to allow script structure check
    class TRUSMoEModel_LargeScale(nn.Module):
        def __init__(self, **kwargs): super().__init__(); self.dummy = nn.Linear(1,1)
        def forward(self, *args, **kwargs): return torch.randn(args[0].size(0), kwargs.get('num_classes', 2)), []
    class TRUSMoEBlock(nn.Module): pass
    def calculate_rus_losses(*args, **kwargs): return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    def calculate_load_balancing_loss(*args, **kwargs): return torch.tensor(0.0)


# --- PyHealth Dataset Wrapper ---
# class PyHealthTRUSDataset(Dataset):
#     """
#     A wrapper for PyHealth datasets to integrate pre-computed RUS values
#     and format data for the TRUSMoEModel_LargeScale.
#     """
#     def __init__(self, pyhealth_dataset, feature_keys: List[str], label_key: str, pad_token: float = 0.0):
#         """
#         Args:
#             pyhealth_dataset: A PyHealth dataset instance (e.g., MIMIC4Dataset after task setup).
#             feature_keys: List of feature keys (modalities) to extract from PyHealth samples.
#                           These correspond to the 'M' dimension.
#             label_key: The key for the target label in PyHealth samples.
#             pad_token: Value used for padding sequences.
#         """
#         self.pyhealth_dataset = pyhealth_dataset
#         self.feature_keys = feature_keys # These are our 'modalities'
#         self.label_key = label_key
#         self.num_modalities = len(feature_keys)
#         self.pad_token = pad_token
#
#         # --- Determine Max Sequence Length ---
#         # This might require iterating through the dataset once or using metadata
#         # For simplicity, we'll assume a fixed max length or use padding/truncation later.
#         # Let's assume PyHealth handles padding/truncation via its dataloader or task function.
#         # We need to know the sequence length 'T' expected by the model.
#
#         # --- !!! CRITICAL: RUS Value Loading !!! ---
#         # This section MUST be adapted to load your pre-computed RUS values.
#         # It needs to align RUS values with the patient_id/visit_id from pyhealth_dataset.
#         # For demonstration, we generate dummy RUS values matching the dataset size.
#         print("WARNING: Generating DUMMY RUS values. Replace with actual RUS loading.")
#         num_samples = len(self.pyhealth_dataset)
#         # Need to know the sequence length T after PyHealth processing
#         # Let's try to get it from the first sample (assuming uniform length after padding)
#         try:
#             sample_data = self.pyhealth_dataset.process_patient(self.pyhealth_dataset.patient_ids[0])
#             # Find the length of a time-varying feature - this is highly dependent on data structure
#             # Example: Assuming 'vitalsign' is a key and has a temporal dimension
#             # This part needs careful inspection of your PyHealth data structure
#             if 'vitalsign' in sample_data[list(sample_data.keys())[0]]: # Check first visit
#                 example_seq_len = len(sample_data[list(sample_data.keys())[0]]['vitalsign'])
#                 print(f"Detected sequence length T = {example_seq_len} from sample data.")
#                 self.seq_len = example_seq_len
#             else:
#                  # Fallback if 'vitalsign' or similar isn't found or structure differs
#                  print("Could not automatically detect sequence length T. Using default or provided arg.")
#                  # Use a default or raise an error - requires user input/config
#                  # For now, let's assume a default or it's passed via args later
#                  self.seq_len = 48 # Example fallback
#                  print(f"Using fallback sequence length T = {self.seq_len}")
#
#         except Exception as e:
#              print(f"Error detecting sequence length: {e}. Using fallback.")
#              self.seq_len = 48 # Example fallback
#              print(f"Using fallback sequence length T = {self.seq_len}")
#
#
#         self.rus_U = torch.rand(num_samples, self.num_modalities, self.seq_len)
#         self.rus_R = torch.rand(num_samples, self.num_modalities, self.num_modalities, self.seq_len)
#         self.rus_S = torch.rand(num_samples, self.num_modalities, self.num_modalities, self.seq_len)
#         self.rus_R = 0.5 * (self.rus_R + self.rus_R.permute(0, 2, 1, 3)) # Symmetrize
#         self.rus_S = 0.5 * (self.rus_S + self.rus_S.permute(0, 2, 1, 3)) # Symmetrize
#         # --- End of RUS Value Loading Placeholder ---
#
#
#     def __len__(self):
#         return len(self.pyhealth_dataset)
#
#     def __getitem__(self, idx):
#         """
#         Retrieves a sample and formats it for the TRUSMoEModel.
#
#         Returns:
#             Tuple: (formatted_data, rus_values, label)
#                    formatted_data: Tensor shape (M, T, E_in)
#                    rus_values: Dict of RUS tensors for this sample
#                    label: Tensor scalar label
#         """
#         # PyHealth dataset typically returns a dictionary for a patient/visit index
#         # Structure depends heavily on the tables and task function used.
#         # Example: sample = {'patient_id': ..., 'visit_id': ..., 'conditions': ..., 'procedures': ..., 'label': ...}
#         pyhealth_sample = self.pyhealth_dataset[idx] # This triggers PyHealth's __getitem__
#
#         # --- Extract Features and Format ---
#         # This is the most dataset-specific part. You need to map the features
#         # defined in `self.feature_keys` to the data structure in `pyhealth_sample`
#         # and arrange them into the (M, T, E_in) tensor.
#         # Assuming E_in=1 (raw values) and features are time-series.
#         # This requires padding/truncation to `self.seq_len`. PyHealth task functions
#         # often handle this, but verify the output format.
#
#         formatted_data_list = []
#         for feature_key in self.feature_keys:
#             # --- Placeholder Logic ---
#             # Find the feature data in pyhealth_sample. This might be nested.
#             # Example: feature_data = pyhealth_sample['vitalsign'][feature_key] # If vitalsign is a dict of lists
#             # Or feature_data = pyhealth_sample[feature_key] # If feature is top-level key
#             # Need to handle missing features/modalities (e.g., pad with `self.pad_token`)
#             # Need to ensure length is `self.seq_len`.
#             # --- --- --- --- --- ---
#             # Dummy feature extraction:
#             # Assume feature data is already a list/tensor of length T
#             # Replace this with actual extraction logic based on PyHealth output
#             if feature_key in pyhealth_sample: # Simplistic check
#                  # Assume the data is directly accessible and needs padding/truncation
#                  raw_feature = pyhealth_sample[feature_key]
#                  # Ensure it's a tensor
#                  if not isinstance(raw_feature, torch.Tensor):
#                      raw_feature = torch.tensor(raw_feature, dtype=torch.float32)
#
#                  # Pad or truncate sequence dimension (T)
#                  current_len = raw_feature.shape[0]
#                  if current_len > self.seq_len:
#                      feature_tensor = raw_feature[:self.seq_len]
#                  elif current_len < self.seq_len:
#                      padding_size = self.seq_len - current_len
#                      # Assume feature dim is last dim or needs unsqueezing if 1D
#                      pad_dims = (0, raw_feature.shape[1] if raw_feature.dim() > 1 else 1) # Pad feature dim
#                      pad_shape = (padding_size,) + raw_feature.shape[1:]
#                      padding = torch.full(pad_shape, self.pad_token, dtype=torch.float32)
#                      feature_tensor = torch.cat((raw_feature, padding), dim=0)
#                  else:
#                      feature_tensor = raw_feature
#
#                  # Ensure feature dimension E_in=1 (or adapt as needed)
#                  if feature_tensor.dim() == 1:
#                      feature_tensor = feature_tensor.unsqueeze(-1) # Add feature dim
#
#             else:
#                 # Handle missing feature: Create a padded tensor
#                 print(f"Warning: Feature '{feature_key}' not found for sample {idx}. Padding.")
#                 feature_tensor = torch.full((self.seq_len, 1), self.pad_token, dtype=torch.float32) # Assuming E_in=1
#
#             # Ensure tensor has shape (T, E_in) before appending
#             # This assumes E_in=1 based on the dummy logic above
#             formatted_data_list.append(feature_tensor[:, :1]) # Take first feature dim if more exist
#
#
#         # Stack features along the modality dimension (M)
#         formatted_data = torch.stack(formatted_data_list, dim=0) # Shape (M, T, E_in)
#
#         # --- Extract Label ---
#         label = torch.tensor(pyhealth_sample[self.label_key], dtype=torch.long) # Ensure correct type for CrossEntropy
#
#         # --- Get RUS Values (using dummy values here) ---
#         rus_values = {
#             'U': self.rus_U[idx], # (M, T)
#             'R': self.rus_R[idx], # (M, M, T)
#             'S': self.rus_S[idx]  # (M, M, T)
#         }
#
#         return formatted_data, rus_values, label

# --- Training and Validation Functions (Modified for PyHealth Dataset) ---

def train_epoch(model: TRUSMoEModel_LargeScale,
                dataloader: DataLoader, # Expects DataLoader yielding (data, rus_values, labels)
                optimizer: optim.Optimizer,
                task_criterion: nn.Module,
                device: torch.device,
                args: argparse.Namespace):
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


    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1}/{args.epochs} [Train]", leave=False)
    for batch_idx, (data, rus_values, labels) in enumerate(progress_bar):
        # Data from PyHealthTRUSDataset should be (B, M, T, E_in)
        data = data.to(device)
        rus_values = {k: v.to(device) for k, v in rus_values.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model expects (B, M, T, E_in)
        final_logits, all_aux_moe_outputs = model(data, rus_values)

        # Calculate Task Loss (assuming classification on aggregated output)
        task_loss = task_criterion(final_logits, labels) # Logits (B, C), Labels (B,)

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
                gating_probs = aux_outputs['gating_probs']
                expert_indices = aux_outputs['expert_indices']
                synergy_expert_indices = layer.moe_layer.synergy_expert_indices
                k = layer.moe_layer.k

                L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                    gating_probs, rus_values, synergy_expert_indices,
                    args.threshold_u, args.threshold_r, args.threshold_s,
                    args.lambda_u, args.lambda_r, args.lambda_s,
                    epsilon=args.epsilon_loss
                )
                L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, args.lambda_load)

                # Check for NaNs in individual aux losses before accumulating
                if not torch.isnan(L_unique): total_L_unique += L_unique
                if not torch.isnan(L_redundancy): total_L_redundancy += L_redundancy
                if not torch.isnan(L_synergy): total_L_synergy += L_synergy
                if not torch.isnan(L_load): total_L_load += L_load


        # Average aux losses
        if num_moe_layers_encountered > 0:
            total_L_unique /= num_moe_layers_encountered
            total_L_redundancy /= num_moe_layers_encountered
            total_L_synergy /= num_moe_layers_encountered
            total_L_load /= num_moe_layers_encountered

        # Combine Losses (check components for NaN before summing)
        loss_components = [task_loss, total_L_unique, total_L_redundancy, total_L_synergy, total_L_load]
        if any(torch.isnan(lc) for lc in loss_components):
             print(f"Warning: NaN detected in loss component at batch {batch_idx}. Skipping backward pass.")
             # Optionally print which loss was NaN
             continue # Skip backward and optimizer step
        else:
             total_loss = sum(loss_components)


        # Backward pass and optimize
        total_loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        optimizer.step()

        # Accumulate losses for logging
        total_loss_accum += total_loss.item()
        task_loss_accum += task_loss.item()
        unique_loss_accum += total_L_unique.item()
        redundancy_loss_accum += total_L_redundancy.item()
        synergy_loss_accum += total_L_synergy.item()
        load_loss_accum += total_L_load.item()

        # Calculate accuracy
        predictions = torch.argmax(final_logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'TaskL': f"{task_loss.item():.4f}",
            'Acc': f"{100. * correct_predictions / total_samples:.2f}%" if total_samples > 0 else "0.00%"
        })

    # Calculate average losses and accuracy for the epoch
    len_dataloader = len(dataloader) if len(dataloader) > 0 else 1 # Avoid division by zero
    avg_total_loss = total_loss_accum / len_dataloader
    avg_task_loss = task_loss_accum / len_dataloader
    avg_unique_loss = unique_loss_accum / len_dataloader
    avg_redundancy_loss = redundancy_loss_accum / len_dataloader
    avg_synergy_loss = synergy_loss_accum / len_dataloader
    avg_load_loss = load_loss_accum / len_dataloader
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    print(f"Epoch {args.current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Aux Losses -> Unique: {avg_unique_loss:.4f}, Redundancy: {avg_redundancy_loss:.4f}, Synergy: {avg_synergy_loss:.4f}, Load: {avg_load_loss:.4f}")

    return avg_total_loss, accuracy


def validate_epoch(model: TRUSMoEModel_LargeScale,
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
        for batch_idx, (data, rus_values, labels) in enumerate(progress_bar):
            data = data.to(device)
            rus_values = {k: v.to(device) for k, v in rus_values.items()}
            labels = labels.to(device)

            final_logits, _ = model(data, rus_values) # Ignore aux outputs for validation loss
            task_loss = task_criterion(final_logits, labels)

            task_loss_accum += task_loss.item()
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix({
                'Val TaskL': f"{task_loss.item():.4f}",
                'Val Acc': f"{100. * correct_predictions / total_samples:.2f}%" if total_samples > 0 else "0.00%"
            })

    len_dataloader = len(dataloader) if len(dataloader) > 0 else 1
    avg_task_loss = task_loss_accum / len_dataloader
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0

    print(f"Epoch {args.current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy


# --- Main Execution ---

def main(args):
    """Main function to set up and run training."""
    # --- Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) # if using multi-GPU
        # Note: Setting deterministic and benchmark flags can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing with PyHealth ---
    print("Loading MIMIC-IV dataset with PyHealth...")
    # Define table scope based on potential task needs
    # Example: Include tables relevant to mortality, LOS, readmission
    tables_to_load = ["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"] # Adjust as needed

    # Simplified mapping for now, assuming tasks handle specifics
    code_mapping = {} # Add specific mappings if required, e.g., NDC to ATC

    mimic4_base_dataset = MIMIC4Dataset(
        root=args.mimic_dir,
        tables=tables_to_load,
        code_mapping=code_mapping,
        dev=args.dev_mode,
        refresh_cache=args.refresh_cache,
    )
    print("Dataset statistics:")
    mimic4_base_dataset.stat()
    # print("Dataset info:") # Can be verbose
    # mimic4_base_dataset.info()

    # --- Task Definition ---
    print(f"Setting up PyHealth task: {args.task_name}...")
    # Dynamically get the task function
    # Make sure the task function names in pyhealth.tasks match these strings
    available_tasks = {
        "mortality_prediction": mortality_prediction_mimic4_fn,
        "length_of_stay_prediction": length_of_stay_prediction_mimic4_fn,
        "readmission_prediction": readmission_prediction_mimic4_fn,
        # Add other tasks here if needed
    }
    if args.task_name not in available_tasks:
        print(f"Error: Task '{args.task_name}' not recognized. Available tasks: {list(available_tasks.keys())}")
        return
    task_fn = available_tasks[args.task_name]

    try:
        # Apply the task function to the base dataset
        # The task function should return a dataset object suitable for splitting
        # (e.g., pyhealth.data.PatientDataset or similar)
        mimic4_task_dataset = mimic4_base_dataset.set_task(task_fn)
        print("Task applied. Sample patient data (first entry):")
        # Print a sample to understand the structure after task processing
        # print(mimic4_task_dataset.samples[0] if mimic4_task_dataset.samples else "No samples found")
        # Note: Accessing .samples might vary depending on PyHealth version/task output

        # --- Feature Keys and Labels ---
        # Define the feature keys your model expects based on the task output
        # This needs to align with how your model consumes the data batch from get_dataloader
        # Example: If task function structures data like {'conditions': [...], 'procedures': [...], 'label': ...}
        # feature_keys = ["conditions", "procedures"] # Adjust based on actual task output structure
        # label_key = "label" # Default label key

        # Determine feature_keys dynamically if possible, or require user input/config
        # Example of trying to infer (may need adjustment based on actual PyHealth dataset structure)
        if hasattr(mimic4_task_dataset, 'feature_keys'):
             feature_keys = mimic4_task_dataset.feature_keys
             print(f"Inferred feature keys: {feature_keys}")
        elif hasattr(mimic4_task_dataset, 'samples') and mimic4_task_dataset.samples:
             # Infer from the first sample, excluding 'patient_id', 'visit_id', 'label'
             sample_keys = list(mimic4_task_dataset.samples[0].keys())
             exclude_keys = ['patient_id', 'visit_id', 'label'] # Common keys to exclude
             feature_keys = [k for k in sample_keys if k not in exclude_keys]
             print(f"Inferred feature keys from sample: {feature_keys}")
        else:
             print("Warning: Could not automatically determine feature keys. You might need to specify them.")
             # Provide a default or raise an error
             # feature_keys = [...] # Define manually if needed
             return # Or handle error

        label_key = "label" # Assume standard 'label' key, adjust if task uses different key

        # --- Determine Sequence Length T ---
        # Try to infer from the dataset/task or use args.seq_len
        # The concept of a single 'seq_len' might be complex if different features have different lengths.
        # PyHealth dataloaders often handle padding/truncation internally based on model needs or defaults.
        # For now, let's pass args.seq_len, but be aware this might not be directly used by get_dataloader
        # in the same way as a custom Dataset's padding.
        effective_seq_len = args.seq_len
        print(f"Using provided --seq_len = {effective_seq_len}. Note: PyHealth's get_dataloader might handle sequence length differently.")


        # --- Create Dataloaders using PyHealth utilities ---
        print("Splitting dataset and creating dataloaders using PyHealth...")
        # Use the mimic4_task_dataset obtained after applying set_task
        train_dataset, val_dataset, test_dataset = split_by_patient(
             mimic4_task_dataset, [0.8, 0.1, 0.1] # Standard 80/10/10 split
        )
        print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        # Create dataloaders using get_dataloader
        # It automatically handles collation and batching for PyHealth datasets
        train_loader = get_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
            # num_workers and pin_memory are not direct args for get_dataloader
            # PyHealth might handle this internally or differently
        )
        val_loader = get_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        # Create test_loader if needed for final evaluation
        # test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print("PyHealth Dataloaders created.")

    except AttributeError as e:
         print(f"Error processing PyHealth dataset or task function output: {e}")
         print("Ensure the task function returns a compatible dataset object and keys are correct.")
         print("Check PyHealth documentation for the expected structure of the task output.")
         # Add more specific debugging info if possible
         # For example, print the type of mimic4_task_dataset
         if 'mimic4_task_dataset' in locals():
             print(f"Type of mimic4_task_dataset: {type(mimic4_task_dataset)}")
         return
    except Exception as e:
         print(f"An unexpected error occurred during data loading/processing: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for debugging
         return


    # --- Model Configuration ---
    # Determine input dimension E_in based on data processing
    # Assuming E_in=1 for now (raw values per feature/modality)
    # Adjust if PyHealth preprocessing creates higher-dimensional features
    actual_input_dim = 1 # Modify based on PyHealth feature format
    print(f"Setting model input dimension E_in = {actual_input_dim}")

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

    # --- Model Instantiation ---
    print("Initializing model...")
    # Calculate max_seq_len for positional encoding
    max_seq_len_model = args.num_modalities * effective_seq_len

    model = TRUSMoEModel_LargeScale(
        input_dim=actual_input_dim, # Use determined input dim
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_modalities=args.num_modalities,
        num_classes=args.num_classes,
        dropout=args.dropout,
        max_seq_len=max_seq_len_model # Use calculated max length
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer and Criterion ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    task_criterion = nn.CrossEntropyLoss() # Assumes classification task

    # --- Training Loop ---
    print("Starting training...")
    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        args.current_epoch = epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, task_criterion, device, args)
        val_loss, val_acc = validate_epoch(model, val_loader, task_criterion, device, args)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_path = os.path.join(args.save_dir, "best_trus_moe_model.pth")
            # Ensure save directory exists
            os.makedirs(args.save_dir, exist_ok=True)
            try:
                 torch.save(model.state_dict(), save_path)
                 print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved to {save_path}")
            except Exception as e:
                 print(f"Error saving model: {e}")


    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

    # --- Add final testing on test set here if needed ---
    # test_loss, test_acc = validate_epoch(model, test_loader, ...)
    # print(f"Final Test Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRUS-MoE Model on MIMIC-IV using PyHealth')

    # Data args
    parser.add_argument('--mimic_dir', type=str, default="/cis/home/xhan56/mimic2.0/physionet.org/files/mimiciv/2.0/hosp", help='Root directory of the MIMIC-IV dataset')
    parser.add_argument('--task_name', type=str, default='mortality_prediction', help='Name of the PyHealth task function (e.g., mortality_prediction_mimic4_fn)')
    # num_modalities will be determined by feature_keys list, but keep for dummy RUS generation size
    parser.add_argument('--num_modalities', type=int, default=10, help='Expected number of features/modalities (used for dummy RUS)')
    # seq_len might be determined by PyHealth task processing, provide as fallback/override
    parser.add_argument('--seq_len', type=int, default=48, help='Expected sequence length T after PyHealth processing, or target length for padding/truncation in wrapper.')
    # input_dim depends on feature representation (e.g., 1 for raw value)
    # parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension per modality/variable (determined after data loading)')
    parser.add_argument('--dev_mode', action='store_true', help='Use PyHealth dev mode (smaller dataset)')
    parser.add_argument('--refresh_cache', action='store_true', help='Refresh PyHealth cache')

    # Model args (mostly same as before)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_moe_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (e.g., 2 for mortality)')

    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8)
    parser.add_argument('--moe_num_synergy_experts', type=int, default=2)
    parser.add_argument('--moe_k', type=int, default=2)
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128)
    # MoE Router specific args
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=64)
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=64)
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=32)
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=32)

    # Training args
    parser.add_argument('--epochs', type=int, default=20) # Increased epochs
    parser.add_argument('--batch_size', type=int, default=32) # Adjusted batch size
    parser.add_argument('--lr', type=float, default=5e-5) # Adjusted LR
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4) # Increased workers
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save best model')


    # Loss args
    parser.add_argument('--threshold_u', type=float, default=0.7)
    parser.add_argument('--threshold_r', type=float, default=0.7)
    parser.add_argument('--threshold_s', type=float, default=0.7)
    parser.add_argument('--lambda_u', type=float, default=0.01)
    parser.add_argument('--lambda_r', type=float, default=0.01)
    parser.add_argument('--lambda_s', type=float, default=0.01)
    parser.add_argument('--lambda_load', type=float, default=0.01)
    parser.add_argument('--epsilon_loss', type=float, default=1e-8)

    args = parser.parse_args()

    # --- Determine max_seq_len based on M and T ---
    # Need to finalize T (seq_len) before calculating this
    # Let's assume T is fixed by args.seq_len for now
    if args.seq_len <= 0:
         print("Error: --seq_len must be positive if not inferred from data.")
         exit()
    args.max_seq_len = args.num_modalities * args.seq_len + 10 # Add buffer

    # --- Determine input_dim ---
    # This should ideally be determined after inspecting the PyHealth processed data
    # For now, we assume it's 1 (raw value per feature)
    args.input_dim = 1
    print(f"Assuming input dimension per modality (E_in) = {args.input_dim}")


    # --- Run Main ---
    args.current_epoch = 0
    main(args)