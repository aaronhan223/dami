# This script computes the RUS of the time series of a stay in MIMIC-IV
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from estimators.ce_alignment_information import (
    CEAlignmentInformation, Discrim, MultimodalDataset,
    train_discrim, eval_discrim, train_ce_alignment, eval_ce_alignment
)

def temporal_pid_label_multi_sequence_batch(X1_list, X2_list, Y_list, X1_masks, X2_masks, lag=1, batch_size=256, n_batches=10, 
                      discrim_epochs=20, ce_epochs=10, seed=42, device=None,
                      hidden_dim=32, layers=2, activation='relu', lr=1e-3, embed_dim=10, n_labels=None,
                      sequence_pooling='timestep', filter_empty_samples=False):
    """
    Compute PID using batch/neural network method for multiple time series sequence/label pairs.
    Parameters:
    -----------
    X1_list, X2_list: List[numpy.ndarray]
        Lists of time series, one per sequence
    Y_list: List[numpy.ndarray]
        Classification labels for each sequence
    X1_masks, X2_masks: List[numpy.ndarray]
        Lists of binary masks indicating valid timesteps for each sequence
    lag: int
        Time lag
    batch_size: int
        Batch size for training
    n_batches : int
        Number of batches to average over
    discrim_epochs : int
        Epochs for discriminator training
    ce_epochs : int
        Epochs for alignment training
    seed : int
        Random seed
    device : torch.device
        Device to run computations on
    hidden_dim : int
        Hidden dimension for neural networks
    layers : int
        Number of layers for neural networks
    activation : str
        Activation function for neural networks
    lr : float
        Learning rate for neural networks
    embed_dim : int
        Embedding dimension for alignment model
    n_labels : int
        Number of class labels. If None, inferred from Y_list
    sequence_pooling : str
        How to process sequences: 'timestep' (default, each timestep as sample), 
        'mean' (mean pooling over masked timesteps only)
    filter_empty_samples : bool
        For timestep mode: whether to filter using masks (default: False, keep all timesteps)
        For mean mode: not applicable (always uses masks)
        
    Returns:
    --------
    results : dict
        PID components
    """
    # Validate parameters
    if len(X1_masks) != len(X1_list):
        raise ValueError(f"X1_masks length ({len(X1_masks)}) must match X1_list length ({len(X1_list)})")
    if len(X2_masks) != len(X2_list):
        raise ValueError(f"X2_masks length ({len(X2_masks)}) must match X2_list length ({len(X2_list)})")
    
    if filter_empty_samples and sequence_pooling != 'timestep':
        raise ValueError("filter_empty_samples=True is only allowed when sequence_pooling='timestep'")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    Y_array = np.array(Y_list)
    unique_labels = np.unique(Y_array)
    actual_n_labels = len(unique_labels)
    if n_labels is None:
        n_labels = actual_n_labels
    else:
        n_labels = max(n_labels, actual_n_labels)
    assert np.all((Y_array >= 0) & (Y_array < n_labels)), f"All labels must be in range [0, {n_labels-1}], but found: {np.unique(Y_array)}" 
    
    # Warn if we have a degenerate case (only one label)
    if actual_n_labels == 1:
        print(f"Warning: Only one unique label found ({unique_labels[0]}). "
              f"This may lead to degenerate distributions. Consider providing multiple classes.")
    
    print(f"Using sequence pooling method: {sequence_pooling}")
    
    # Collect all data points
    all_X1_data = []
    all_X2_data = []
    all_Y_labels = []
    
    # Track filtering statistics for timestep mode
    total_timesteps = 0
    filtered_timesteps = 0

    for i, (X1, X2, Y) in enumerate(zip(X1_list, X2_list, Y_list)):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)

        if len(X1) != len(X2):
            raise ValueError("X1 and X2 must have the same length")

        if lag == 0:
            X1_past = X1
            X2_past = X2
        else:
            X1_past = X1[:-lag, :]
            X2_past = X2[:-lag, :]
        
        if len(X1_past) == 0:
            continue

        # Get masks for this sequence
        X1_mask = X1_masks[i]
        X2_mask = X2_masks[i]
        
        # Apply lag to masks as well
        if lag == 0:
            X1_mask_past = X1_mask
            X2_mask_past = X2_mask
        else:
            X1_mask_past = X1_mask[:-lag]
            X2_mask_past = X2_mask[:-lag]

        if sequence_pooling == 'timestep':
            # Treat each time step as independent data point
            if filter_empty_samples:
                # Filter using masks - only keep timesteps where both X1 and X2 have valid data
                valid_mask = X1_mask_past & X2_mask_past
                
                total_timesteps += len(X1_past)
                n_valid = np.sum(valid_mask)
                filtered_timesteps += n_valid
                
                if n_valid > 0:
                    X1_filtered = X1_past[valid_mask]
                    X2_filtered = X2_past[valid_mask]
                    all_X1_data.extend(X1_filtered)
                    all_X2_data.extend(X2_filtered)
                    all_Y_labels.extend([Y] * len(X1_filtered))
                # If no valid timesteps, skip this sequence entirely
            else:
                # Keep all timesteps regardless of mask
                all_X1_data.extend(X1_past)
                all_X2_data.extend(X2_past)
                all_Y_labels.extend([Y] * len(X1_past))
        elif sequence_pooling == 'mean':
            # Mean pooling approach: only average over masked (valid) timesteps
            X1_valid_mask = X1_mask_past
            X2_valid_mask = X2_mask_past
            
            if np.any(X1_valid_mask) and np.any(X2_valid_mask):
                X1_pooled = np.mean(X1_past[X1_valid_mask], axis=0)
                X2_pooled = np.mean(X2_past[X2_valid_mask], axis=0)
                all_X1_data.append(X1_pooled)
                all_X2_data.append(X2_pooled)
                all_Y_labels.append(Y)
            # If no valid timesteps in either modality, skip this sequence
        else:
            raise ValueError(f"Unknown sequence_pooling method: {sequence_pooling}. Choose from 'timestep', 'mean'")
    
    all_X1_data = np.array(all_X1_data)
    all_X2_data = np.array(all_X2_data)
    all_Y_labels = np.array(all_Y_labels)

    # Print filtering statistics for timestep mode
    if sequence_pooling == 'timestep' and filter_empty_samples:
        n_removed = total_timesteps - filtered_timesteps
        if total_timesteps > 0:
            print(f"Mask filtering: Kept {filtered_timesteps}/{total_timesteps} timesteps ({filtered_timesteps/total_timesteps*100:.1f}%), removed {n_removed} invalid timesteps")
        else:
            print("Mask filtering: No timesteps to process")

    # Check if we have any data after processing
    if len(all_X1_data) == 0:
        raise ValueError("No valid data found after processing! Consider adjusting filter_empty_samples or check your input data.")

    X1_tensor = torch.tensor(all_X1_data, dtype=torch.float32, device=device)
    X2_tensor = torch.tensor(all_X2_data, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(all_Y_labels, dtype=torch.long, device=device)

    X1_tensor = X1_tensor.view(-1, X1_tensor.shape[-1])
    X2_tensor = X2_tensor.view(-1, X2_tensor.shape[-1])
    Y_tensor = Y_tensor.view(-1)
    
    print(f"Using sequence pooling method: {sequence_pooling}")
    print(f"Final tensor shapes: X1={X1_tensor.shape}, X2={X2_tensor.shape}, Y={Y_tensor.shape}")

    # Create train/test split
    n_samples = len(Y_tensor)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_ds = MultimodalDataset(
        [X1_tensor[train_idx], X2_tensor[train_idx]], 
        Y_tensor[train_idx]
    )
    test_ds = MultimodalDataset(
        [X1_tensor[test_idx], X2_tensor[test_idx]], 
        Y_tensor[test_idx]
    )
    
    all_results = []
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        # Sample subset of data for this batch
        if len(train_ds) > batch_size * 10:
            batch_indices = np.random.choice(len(train_ds), 
                                           size=min(batch_size * 10, len(train_ds)), 
                                           replace=False)
            batch_X1 = X1_tensor[train_idx[batch_indices]]
            batch_X2 = X2_tensor[train_idx[batch_indices]]
            batch_Y = Y_tensor[train_idx[batch_indices]]
            
            batch_train_ds = MultimodalDataset([batch_X1, batch_X2], batch_Y)
        else:
            batch_train_ds = train_ds
            
        # Train discriminators
        x1_dim = X1_tensor.shape[1]
        x2_dim = X2_tensor.shape[1]
        
        model_discrim_1 = Discrim(x_dim=x1_dim, hidden_dim=hidden_dim, num_labels=n_labels, 
                                 layers=layers, activation=activation).to(device)
        model_discrim_2 = Discrim(x_dim=x2_dim, hidden_dim=hidden_dim, num_labels=n_labels, 
                                 layers=layers, activation=activation).to(device)
        model_discrim_12 = Discrim(x_dim=x1_dim + x2_dim, hidden_dim=hidden_dim, 
                                  num_labels=n_labels, layers=layers, activation=activation).to(device)
        
        # Train each discriminator
        for model, data_type in [
            (model_discrim_1, ([1], [0])),
            (model_discrim_2, ([2], [0])),
            (model_discrim_12, ([1], [2], [0])),
        ]:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_loader = DataLoader(batch_train_ds, shuffle=True, 
                                    batch_size=min(batch_size, len(batch_train_ds)),
                                    num_workers=0)
            train_discrim(model, train_loader, optimizer, 
                         data_type=data_type, num_epoch=discrim_epochs, device=device)
            model.eval()
    
        # Compute prior P(Y)
        p_y = torch.zeros(n_labels)
        for i in range(n_labels):
            p_y[i] = (Y_tensor[train_idx] == i).float().mean()
        p_y = p_y.to(device)
        
        # Create alignment model
        model = CEAlignmentInformation(
            x1_dim=x1_dim, x2_dim=x2_dim,
            hidden_dim=hidden_dim, embed_dim=embed_dim, num_labels=n_labels, 
            layers=layers, activation=activation,
            discrim_1=model_discrim_1, discrim_2=model_discrim_2, 
            discrim_12=model_discrim_12, p_y=p_y
        ).to(device)
        
        opt_align = optim.Adam(model.align_parameters(), lr=lr)
        
        # Train alignment
        train_loader = DataLoader(batch_train_ds, shuffle=True, 
                                batch_size=min(batch_size, len(batch_train_ds)),
                                num_workers=0)
        model.train()
        train_ce_alignment(model, train_loader, opt_align, 
                        data_type=([1], [2], [0]), num_epoch=ce_epochs, device=device)
        
        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(test_ds, shuffle=False, 
                            batch_size=min(batch_size, len(test_ds)),
                            num_workers=0)
        results, _ = eval_ce_alignment(model, test_loader, data_type=([1], [2], [0]), device=device)
        
        # Average results across batches within this iteration
        batch_result = torch.mean(results, dim=0).cpu().numpy() / np.log(2)  # Convert to bits
        all_results.append(batch_result)
        
    # Average across all batches
    avg_results = np.mean(all_results, axis=0)

    return {
        'redundancy': max(0, avg_results[0]),
        'unique_x1': max(0, avg_results[1]),
        'unique_x2': max(0, avg_results[2]),
        'synergy': max(0, avg_results[3]),
        'total_di': sum(max(0, x) for x in avg_results),
        'method': 'batch'
    }

def align_multimodal_irg_ts(reg_ts: np.ndarray, multimodal_irg_times_feats: Dict[str, List[Tuple[float, np.ndarray]]], modality_dim_dict: Dict[str, int], interval_length=1):
    """
    Align the regular time series (labs + vitals) with multimodal irrregular time series features (notes + cxr).
    Args:
        reg_ts: The regular time series, shape: (T, D). T is the number of time steps, D is the number of features.
        multimodal_irg_times_feats: A dictionary of multimodal irregular time series features.
            The key is the modality name, the value is a list of tuples, each containing the timestamp and the feature vector.
        modality_dim_dict: A dictionary of modality names and their dimensions.
        interval_length: The length of the intervals between the regular time series time steps for us to plug in the multimodal features.
    Returns:
        A dictionary of aligned time series and masks. The key is the modality name, the value is a tuple of:
        - aligned time series of shape: (T, number of features in the modality)
        - binary mask of shape: (T,) indicating which timesteps have actual irregular data (1) vs zero-filled (0)
    """
    num_time_steps = len(reg_ts)
    multimodal_reg_ts = {}
    
    for modality, feats_list in multimodal_irg_times_feats.items():
        # Get number of features for this modality
        num_features = modality_dim_dict[modality]
        aligned_ts = np.zeros((num_time_steps, num_features))
        mask = np.zeros(num_time_steps, dtype=bool)  # Binary mask for valid timesteps
        
        # Handle empty feature lists
        if not feats_list:
            print(f"Warning: No features found for modality {modality}, using all zeros")
            # Keep aligned_ts as all zeros and mask as all False
            multimodal_reg_ts[modality] = (aligned_ts, mask)
            continue
        
        # First pass: collect all features by index
        features_by_index = {}
        for time, feats in feats_list:
            index = int(time / interval_length)
            
            # Skip if index is out of bounds
            if index >= num_time_steps or index < 0:
                print(f"Warning: Time {time} maps to index {index} which is out of bounds {num_time_steps} for modality {modality}")
                continue
                
            if index not in features_by_index:
                features_by_index[index] = []
            features_by_index[index].append(feats)
        
        # Second pass: average features at each index and set mask
        for index, feat_list in features_by_index.items():
            # if len(feat_list) > 1:
            #     print(f"Averaging {len(feat_list)} measurements at index {index} for modality {modality}")
            aligned_ts[index, :] = np.mean(feat_list, axis=0)
            mask[index] = True  # Mark this timestep as having actual data
            
        multimodal_reg_ts[modality] = (aligned_ts, mask)
        
    return multimodal_reg_ts

def preprocess_mimiciv_data(stays: List[Dict], num_subsample_stays: int = None) -> List[Dict]:
    """
    Preprocess the MIMIC-IV data.
    Args:
        stays: List of stays.
        num_subsample_stays: Number of stays to randomly sample for analysis. If None, use all eligible stays.
    Returns:
        all_multimodal_reg_ts: List of multimodal regular time series.
        all_labels: List of labels.
    """
    eligible_stays = []
    for stay in stays:
        if len(stay['ts_tt']) > 12:
            eligible_stays.append(stay)
    
    print(f"Found {len(eligible_stays)} eligible stays (more than 12 labs/vitals record times)")
    
    # Randomly sample num_subsample_stays from eligible stays
    if num_subsample_stays is None:
        print("Using all eligible stays for analysis.")
        selected_stays = eligible_stays
    elif num_subsample_stays > len(eligible_stays):
        print(f"Warning: Requested {num_subsample_stays} stays but only {len(eligible_stays)} are eligible. Using all eligible stays.")
        selected_stays = eligible_stays
    else:
        selected_stays = np.random.choice(eligible_stays, size=num_subsample_stays, replace=False).tolist()
    
    print(f"Selected {len(selected_stays)} stays for analysis")

    # Process all selected stays

    modality_dim_dict = {'labs_vitals': 30,
                         'cxr': 1024,
                         'notes': 768}
    all_multimodal_reg_ts = []
    all_labels = []
    
    for i, stay in enumerate(selected_stays):
        
        multimodal_irg_ts = {
            'notes': [(stay['text_time'][j], stay['text_embeddings'][j]) for j in range(len(stay['text_time']))],
            'cxr': [(stay['cxr_time'][j], stay['cxr_feats'][j]) for j in range(len(stay['cxr_time']))]
        }
        
        multimodal_reg_ts = align_multimodal_irg_ts(stay['reg_ts'], multimodal_irg_ts, modality_dim_dict)
        # Add labs_vitals with a full mask (all timesteps are valid for regular time series)
        multimodal_reg_ts['labs_vitals'] = (stay['reg_ts'], np.ones(len(stay['reg_ts']), dtype=bool))
        
        all_multimodal_reg_ts.append(multimodal_reg_ts)
        all_labels.append(stay['label'])
    
    return all_multimodal_reg_ts, all_labels

def main(args):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    # Randomly sample num_subsample_stays from eligible stays
    np.random.seed(args.seed)

    train_stays = pickle.load(open(args.train_dataset_path, 'rb'))
    
    all_multimodal_reg_ts, all_labels = preprocess_mimiciv_data(train_stays, args.num_subsample_stays)    
    
    # Get modality names from first stay (assuming all stays have same modalities)
    modality_names = list(all_multimodal_reg_ts[0].keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))
    print(f"\nGenerated {len(modality_pairs)} pairs of modalities for analysis: {modality_pairs}")
    
    dominant_pid_results = []
    all_pid_results = []
    
    # Prepare data for temporal_pid_label_multi_sequence_batch
    for mod1, mod2 in modality_pairs:
        print(f"\n--- Analyzing Modality Pair {mod1} vs {mod2} ---")
        
        # Extract time series for this modality pair across all stays
        # Each modality now returns (time_series, mask) tuple, so extract just the time series
        X1_list = [stay_data[mod1][0] for stay_data in all_multimodal_reg_ts]  # Extract time series
        X2_list = [stay_data[mod2][0] for stay_data in all_multimodal_reg_ts]  # Extract time series
        Y_list = all_labels
        
        # Also extract masks for potential future use
        X1_masks = [stay_data[mod1][1] for stay_data in all_multimodal_reg_ts]  # Extract masks
        X2_masks = [stay_data[mod2][1] for stay_data in all_multimodal_reg_ts]  # Extract masks
        
        print(f"Number of stays: {len(X1_list)}")
        print(f"X1 ({mod1}) shapes: {[x.shape for x in X1_list[:3]]}{'...' if len(X1_list) > 3 else ''}")
        print(f"X2 ({mod2}) shapes: {[x.shape for x in X2_list[:3]]}{'...' if len(X2_list) > 3 else ''}")
        print(f"Labels: {Y_list[:10]}{'...' if len(Y_list) > 10 else ''}")
        
        pid_results = temporal_pid_label_multi_sequence_batch(
            X1_list, X2_list, Y_list, X1_masks, X2_masks,
            lag=0, # always set lag to 0 because y is a static target
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            discrim_epochs=args.discrim_epochs,
            ce_epochs=args.ce_epochs,
            seed=args.seed,
            device=device,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            activation=args.activation,
            lr=args.lr, # use 1e-2 because the default lr is too low and the loss decreases too slowly
            embed_dim=args.embed_dim,
            n_labels=len(np.unique(Y_list)),
            sequence_pooling=args.sequence_pooling,
            filter_empty_samples=args.filter_empty
        )
        print(f"PID Results for {mod1} vs {mod2}: {pid_results}")

        r = pid_results['redundancy']
        u1 = pid_results['unique_x1']
        u2 = pid_results['unique_x2']
        s = pid_results['synergy']
        mi = pid_results['total_di']

        if mi > 1e-9:
            r_norm = r / mi
            u1_norm = u1 / mi
            u2_norm = u2 / mi
            s_norm = s / mi


            results = {
                'R_value': r,
                'U1_value': u1,
                'U2_value': u2,
                'S_value': s,
                'MI_value': mi,
                'R_norm': r_norm,
                'U1_norm': u1_norm,
                'U2_norm': u2_norm,
                'S_norm': s_norm
            }

            norm_values = {
                'R': r_norm,
                'U1': u1_norm,
                'U2': u2_norm,
                'S': s_norm
            }
            max_term = None
            max_value = -1
            for term, value in norm_values.items():
                if value > max_value:
                    max_value = value
                    max_term = term
            if max_term and max_value > args.dominance_threshold:
                print(f"Dominant term for {mod1} vs {mod2}: {max_term} with value {max_value}")
                dominant_pid_results.append({
                    'feature_pair': (mod1, mod2),
                    'dominant_term': max_term,
                    'dominance_ratio': 1, # always set to 1 because we only analyzed one lag
                    'lags_analyzed': 1,
                    'avg_metrics': results,
                    'lag_results': [results],
                    # 'modality1_features': modality_names[mod1],
                    # 'modality2_features': modality_names[mod2],
                    'n_features_mod1': X1_list[0].shape[1],
                    'n_features_mod2': X2_list[0].shape[1]
                })


            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': results,
                'lag_results': [results],
                # 'modality1_features': modality_names[mod1],
                # 'modality2_features': modality_names[mod2],
                'n_features_mod1': X1_list[0].shape[1],
                'n_features_mod2': X2_list[0].shape[1]
            })

    if dominant_pid_results:
        output_filename = f'mimiciv_rus_multimodal_dominant_thresh{args.dominance_threshold:.1f}.npy'
        output_path = os.path.join(args.output_dir, output_filename)
        print(f"Saving {len(dominant_pid_results)} dominant multimodal PID results to {output_path}...")
        np.save(output_path, dominant_pid_results, allow_pickle=True)
        print("Saving dominant modality pairs complete.")
    
    if all_pid_results:
        all_output_filename = f'mimiciv_rus_multimodal_all.npy'
        all_output_path = os.path.join(args.output_dir, all_output_filename)
        print(f"Saving {len(all_pid_results)} PID results for all modality pairs to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True)
        print("Saving all modality pairs complete.")

    # --- Print summary of dominant terms ---
    if dominant_pid_results:
        dominance_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        for result in dominant_pid_results:
            term = result.get('dominant_term')
            if term in dominance_counts:
                dominance_counts[term] += 1

        print("\n--- Multimodal Dominance Summary ---")
        print(f"Total modality pairs with dominant terms: {len(dominant_pid_results)}")
        for term, count in dominance_counts.items():
            print(f"  {term} dominant: {count} pairs")
        print("----------------------------------------")

    else:
        print("No dominant PID terms found with the current threshold and percentage criteria.")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIMIC-IV RUS computation')
    parser.add_argument('--output_dir', type=str, default='../results/mimiciv',
                        help='Directory to save analysis results')
    parser.add_argument('--train_dataset_path', type=str, required=True,
                        help='Path to the preprocessed MIMIC-IV train dataset')
    parser.add_argument('--num_subsample_stays', type=int, default=None,
                        help='Number of stays to randomly sample for analysis')
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Maximum lag for temporal PID analysis')
    parser.add_argument('--bins', type=int, default=4,
                        help='Number of bins for discretization (reduced for multivariate)')
    parser.add_argument('--dominance_threshold', type=float, default=0.4,
                        help='Threshold for a PID term to be considered dominant')
    parser.add_argument('--dominance_percentage', type=float, default=0.9,
                        help='Percentage of lags a term must dominate to be considered dominant overall')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'joint', 'cvxpy', 'batch'],
                        help='PID estimation method (auto: choose based on dimensionality)')
    # General parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for batch method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    
    # BATCH method specific parameters
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches for batch method')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension for neural networks in batch method')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers for neural networks in batch method')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'],
                        help='Activation function for neural networks in batch method')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for neural networks in batch method')
    parser.add_argument('--embed_dim', type=int, default=10,
                        help='Embedding dimension for alignment model in batch method')
    parser.add_argument('--discrim_epochs', type=int, default=100,
                        help='Number of epochs for discriminator training in batch method')
    parser.add_argument('--ce_epochs', type=int, default=10,
                        help='Number of epochs for CE alignment training in batch method')
    parser.add_argument('--sequence_pooling', type=str, default='timestep',
                        choices=['timestep', 'mean'],
                        help='How to process sequences for batch method (timestep, mean)')
    
    # Data filtering parameters
    parser.add_argument('--filter_empty', action='store_true',
                        help='For timestep mode: filter timesteps using masks (default: keep all timesteps)')
    
    # CVXPY method specific parameters
    parser.add_argument('--regularization', type=float, default=1e-6,
                        help='Regularization parameter for CVXPY method')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)