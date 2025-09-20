#!/usr/bin/env python3
"""
Full PAMAP2 Experiment: Temporal RUS Computation + Multimodal TRUS MoE Training

This script performs:
1. Temporal RUS computation using BATCH estimator (pamap_rus_multimodal.py)
2. Multimodal TRUS MoE training (train_pamap_multimodal.py)

Key requirements:
- Same train/val/test split for both BATCH estimator and TRUS MoE model
- Compute temporal RUS separately for train/val/test splits
- Use training RUS for model training, validation RUS for validation, test RUS for testing
"""

import pandas as pd
import numpy as np
import math
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
import random
import wandb
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional, Set
import warnings
import pdb

# Import required modules
try:
    from temporal_pid_multivariate import multi_lag_analysis
    from pamap_rus_multimodal import (
        get_pamap_column_names, load_pamap_data, preprocess_pamap_data, 
        categorize_pamap_sensors
    )
    from train_pamap_multimodal import (
        MultimodalPamapDataset, categorize_pamap_sensors as categorize_sensors_train,
        load_multimodal_rus_data, collate_multimodal,
        train_epoch_multimodal, validate_epoch_multimodal
    )
    from trus_moe_multimodal import MultimodalTRUSMoEModel
    from trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
    from plots.plot_expert_activation import analyze_expert_activations
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)


DEFAULT_DATASET_DIR = "/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol"
DEFAULT_OUTPUT_DIR = "../results/pamap_full_exp"


def load_all_subjects_data(dataset_dir, subject_list):
    """Load and combine data from multiple subjects"""
    print(f"Loading data for subjects: {subject_list}")

    all_data = []
    for subject_id in subject_list:
        try:
            df = load_pamap_data(subject_id, dataset_dir)
            df_processed, sensor_columns = preprocess_pamap_data(df)

            if not df_processed.empty:
                # Add subject_id column to track which subject data comes from
                df_processed['subject_id'] = subject_id
                all_data.append(df_processed)
                print(f"Subject {subject_id}: {df_processed.shape[0]} samples")
            else:
                print(f"Warning: No data for subject {subject_id} after preprocessing")
        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data loaded from any subject")

    # Combine all subjects
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")

    # Get sensor columns from the first processed dataframe
    sensor_columns = [col for col in combined_df.columns if col not in ['activity_id', 'subject_id']]

    return combined_df, sensor_columns


def create_subject_based_splits(combined_df, train_subjects, val_subjects, test_subjects):
    """Create train/val/test splits based on subject IDs"""
    train_data = combined_df[combined_df['subject_id'].isin(train_subjects)].copy()
    val_data = combined_df[combined_df['subject_id'].isin(val_subjects)].copy()
    test_data = combined_df[combined_df['subject_id'].isin(test_subjects)].copy()

    # Keep subject_id column for proper lagged data preparation
    # It will be used in prepare_lagged_data_per_subject and then handled in dataset creation
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(f"Subject-based splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data


def parse_args():
    parser = argparse.ArgumentParser(description='Full PAMAP2 Experiment: RUS Computation + TRUS MoE Training')

    # Data args
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                        help='Directory containing PAMAP2 dataset files')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save analysis results')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='Subject ID to analyze (1-9) - DEPRECATED: now uses all subjects')
    parser.add_argument('--use_all_subjects', action='store_true', default=True,
                        help='Use all subjects with predefined train/val/test splits')
    
    # Data splitting args (DEPRECATED when use_all_subjects=True)
    parser.add_argument('--train_split', type=float, default=0.6,
                        help='Training data split fraction (unused when use_all_subjects=True)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation data split fraction (unused when use_all_subjects=True)')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Test data split fraction (unused when use_all_subjects=True)')

    # Subject-based splits (when use_all_subjects=True)
    parser.add_argument('--train_subjects', type=str, default='1,2,3,4,5,6',
                        help='Comma-separated list of training subject IDs')
    parser.add_argument('--val_subjects', type=str, default='7',
                        help='Comma-separated list of validation subject IDs')
    parser.add_argument('--test_subjects', type=str, default='8,9',
                        help='Comma-separated list of test subject IDs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # RUS computation args (BATCH estimator)
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Maximum lag for temporal PID analysis')
    parser.add_argument('--bins', type=int, default=4,
                        help='Number of bins for discretization')
    parser.add_argument('--dominance_threshold', type=float, default=0.4,
                        help='Threshold for a PID term to be considered dominant')
    parser.add_argument('--dominance_percentage', type=float, default=0.9,
                        help='Percentage of lags a term must dominate to be considered dominant overall')
    
    # BATCH method specific parameters
    parser.add_argument('--batch_size_rus', type=int, default=512,
                        help='Batch size for BATCH method')
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches for batch method')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension for neural networks in batch method')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers for neural networks in batch method')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'],
                        help='Activation function for neural networks in batch method')
    parser.add_argument('--lr_rus', type=float, default=1e-3,
                        help='Learning rate for neural networks in batch method')
    parser.add_argument('--embed_dim', type=int, default=10,
                        help='Embedding dimension for alignment model in batch method')
    parser.add_argument('--discrim_epochs', type=int, default=20,
                        help='Number of epochs for discriminator training in batch method')
    parser.add_argument('--ce_epochs', type=int, default=10,
                        help='Number of epochs for CE alignment training in batch method')
    
    # TRUS MoE Model args
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length (T)')
    parser.add_argument('--window_step', type=int, default=50, help='Step size for sliding window')
    
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
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
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
    
    # System args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--cuda_device', type=int, default=0, help='Specific GPU to use')
    
    # Distributed training args
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    
    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='pamap-full-exp',
                        help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')
    
    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true',
                        help='Generate expert activation plots after training')
    parser.add_argument('--plot_num_samples', type=int, default=32,
                        help='Number of samples to use for expert activation plotting')
    
    return parser.parse_args()


class MultimodalDataSplitter:
    """Helper class to maintain consistent data splits between RUS computation and TRUS MoE training"""
    
    def __init__(self, df_processed, sensor_columns, args):
        self.df_processed = df_processed
        self.sensor_columns = sensor_columns
        self.args = args
        self.modality_sensors = categorize_pamap_sensors(sensor_columns)
        
        # Ensure splits sum to 1.0
        total_split = args.train_split + args.val_split + args.test_split
        if abs(total_split - 1.0) > 1e-6:
            print(f"Warning: Splits do not sum to 1.0 ({total_split}). Normalizing...")
            args.train_split /= total_split
            args.val_split /= total_split
            args.test_split /= total_split
        
        # Create indices for train/val/test splits
        self._create_splits()
    
    def _create_splits(self):
        """Create consistent train/val/test splits"""
        total_samples = len(self.df_processed)
        
        # Set random seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        
        # Create shuffled indices
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(total_samples * self.args.train_split)
        val_size = int(total_samples * self.args.val_split)
        test_size = total_samples - train_size - val_size
        
        # Split indices
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:train_size + val_size]
        self.test_indices = indices[train_size + val_size:]
        
        print(f"Data split sizes: Train={len(self.train_indices)}, Val={len(self.val_indices)}, Test={len(self.test_indices)}")
    
    def get_split_data(self, split='train'):
        """Get data for a specific split"""
        if split == 'train':
            indices = self.train_indices
        elif split == 'val':
            indices = self.val_indices
        elif split == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Get data for this split
        split_df = self.df_processed.iloc[indices].copy()
        
        # Reset index
        split_df = split_df.reset_index(drop=True)
        
        return split_df, indices


def prepare_lagged_data_per_subject(split_data, modality_sensors, max_lag):
    """Prepare lagged data by processing each subject individually to prevent cross-subject time mixing"""
    # Get all unique subject IDs from the split data (if subject_id column exists)
    if 'subject_id' in split_data.columns:
        split_subject_ids = split_data['subject_id'].unique()
    else:
        # If no subject_id column, treat as single subject
        split_subject_ids = [None]

    # Generate pairs of modalities
    modality_names = list(modality_sensors.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))

    # Prepare combined data for each modality pair
    combined_data = {}

    for mod1, mod2 in modality_pairs:
        combined_X1 = []
        combined_X2 = []
        combined_Y = []

        for subject_id in split_subject_ids:
            if subject_id is not None:
                subject_data = split_data[split_data['subject_id'] == subject_id]
            else:
                subject_data = split_data

            if len(subject_data) <= max_lag:
                print(f"Warning: Subject {subject_id} has insufficient data ({len(subject_data)} <= {max_lag})")
                continue

            # Prepare modality data for this subject
            subject_mod1_sensors = [s for s in modality_sensors[mod1] if s in subject_data.columns]
            subject_mod2_sensors = [s for s in modality_sensors[mod2] if s in subject_data.columns]

            if not subject_mod1_sensors or not subject_mod2_sensors:
                print(f"Warning: Missing sensors for subject {subject_id}")
                continue

            X1_subject = subject_data[subject_mod1_sensors].values
            X2_subject = subject_data[subject_mod2_sensors].values
            Y_subject = subject_data['activity_label'].values

            # Add subject data to combined arrays
            combined_X1.append(X1_subject)
            combined_X2.append(X2_subject)
            combined_Y.append(Y_subject)

        # Store combined data for this modality pair
        if combined_X1:
            combined_data[(mod1, mod2)] = {
                'X1': combined_X1,
                'X2': combined_X2,
                'Y': combined_Y
            }

    return combined_data


def compute_temporal_rus_for_split_multi_subject(split_data, modality_sensors, args, split_name, device, subject_ids):
    """Compute temporal RUS for a specific data split with proper per-subject lagged data preparation"""
    print(f"\n=== Computing temporal RUS for {split_name} split (subjects: {subject_ids}) ===")

    # Determine unique activities and create mapping
    unique_activities = sorted([act for act in split_data['activity_id'].unique() if act != 0])
    activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}

    # Remap activity IDs
    split_data['activity_label'] = split_data['activity_id'].map(activity_map)
    split_data.dropna(subset=['activity_label'], inplace=True)
    split_data['activity_label'] = split_data['activity_label'].astype(int)

    # Prepare combined data for multi-subject analysis
    combined_data = prepare_lagged_data_per_subject(split_data, modality_sensors, args.max_lag)

    if not combined_data:
        print(f"Warning: No valid data for {split_name} split")
        return None, activity_map

    all_pid_results = []

    for i, ((mod1, mod2), data) in enumerate(combined_data.items()):
        print(f"\n--- Analyzing Modality Pair {i+1}/{len(combined_data)}: {mod1} vs {mod2} ---")

        X1_list = data['X1']
        X2_list = data['X2']
        Y_list = data['Y']

        try:
            # Create a custom multi_lag_analysis that handles per-subject lagging
            pid_results = multi_lag_analysis_multi_subject(
                X1_list, X2_list, Y_list,
                max_lag=args.max_lag, bins=args.bins,
                method='batch',
                batch_size=min(args.batch_size_rus, sum(len(y) for y in Y_list)//2),
                n_batches=args.n_batches,
                seed=args.seed,
                device=device,
                hidden_dim=args.hidden_dim,
                layers=args.layers,
                activation=args.activation,
                lr=args.lr_rus,
                embed_dim=args.embed_dim,
                discrim_epochs=args.discrim_epochs,
                ce_epochs=args.ce_epochs
            )
        except Exception as e:
            print(f"Error during PID analysis for pair ({mod1}, {mod2}): {e}")
            continue

        # Process results
        lags = pid_results.get('lag', range(args.max_lag + 1))
        lag_results = []

        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx]
                u2 = pid_results['unique_x2'][lag_idx]
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                if mi > 1e-9:
                    lag_results.append({
                        'lag': lag,
                        'R_value': r,
                        'U1_value': u1,
                        'U2_value': u2,
                        'S_value': s,
                        'MI_value': mi,
                        'R_norm': r / mi,
                        'U1_norm': u1 / mi,
                        'U2_norm': u2 / mi,
                        'S_norm': s / mi
                    })
            except (IndexError, KeyError) as e:
                print(f"Warning: Error processing lag {lag} for pair ({mod1}, {mod2}): {e}")
                continue

        if lag_results:
            # Calculate average metrics across all lags
            avg_metrics = {
                'R_value': np.mean([r['R_value'] for r in lag_results]),
                'U1_value': np.mean([r['U1_value'] for r in lag_results]),
                'U2_value': np.mean([r['U2_value'] for r in lag_results]),
                'S_value': np.mean([r['S_value'] for r in lag_results]),
                'MI_value': np.mean([r['MI_value'] for r in lag_results]),
                'R_norm': np.mean([r['R_norm'] for r in lag_results]),
                'U1_norm': np.mean([r['U1_norm'] for r in lag_results]),
                'U2_norm': np.mean([r['U2_norm'] for r in lag_results]),
                'S_norm': np.mean([r['S_norm'] for r in lag_results])
            }

            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': avg_metrics,
                'lag_results': lag_results,
                'modality1_features': modality_sensors[mod1],
                'modality2_features': modality_sensors[mod2],
                'n_features_mod1': len(modality_sensors[mod1]),
                'n_features_mod2': len(modality_sensors[mod2])
            })

    print(f"\nAnalysis complete for {split_name} split: {len(all_pid_results)} modality pairs processed.")

    return all_pid_results, activity_map


def multi_lag_analysis_multi_subject(X1_list, X2_list, Y_list, max_lag, bins, **kwargs):
    """
    Modified multi_lag_analysis that handles multiple subjects by creating
    lagged datasets for each subject individually, then concatenating
    """
    results = {
        'lag': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': []
    }

    for lag in range(max_lag + 1):
        # Prepare lagged data for each subject
        combined_X1_lag = []
        combined_X2_lag = []
        combined_Y_lag = []

        for X1_subj, X2_subj, Y_subj in zip(X1_list, X2_list, Y_list):
            if lag > 0:
                if len(Y_subj) <= lag:
                    continue
                X1_lagged = X1_subj[:-lag]
                X2_lagged = X2_subj[:-lag]
                Y_lagged = Y_subj[lag:]
            else:
                X1_lagged = X1_subj
                X2_lagged = X2_subj
                Y_lagged = Y_subj

            if len(Y_lagged) > 0:
                combined_X1_lag.append(X1_lagged)
                combined_X2_lag.append(X2_lagged)
                combined_Y_lag.append(Y_lagged)

        if not combined_X1_lag:
            continue

        # Concatenate lagged data from all subjects
        X1_lag = np.vstack(combined_X1_lag)
        X2_lag = np.vstack(combined_X2_lag)
        Y_lag = np.concatenate(combined_Y_lag)

        # Call single lag analysis
        try:
            lag_result = multi_lag_analysis(X1_lag, X2_lag, Y_lag, max_lag=0, bins=bins, **kwargs)

            # Extract results for lag 0 (which is our actual lag)
            results['lag'].append(lag)
            results['redundancy'].append(lag_result['redundancy'][0])
            results['unique_x1'].append(lag_result['unique_x1'][0])
            results['unique_x2'].append(lag_result['unique_x2'][0])
            results['synergy'].append(lag_result['synergy'][0])
            results['total_di'].append(lag_result['total_di'][0])

        except Exception as e:
            print(f"Error computing PID for lag {lag}: {e}")
            continue

    return results


def compute_temporal_rus_for_split(split_data, modality_sensors, args, split_name, device):
    """Compute temporal RUS for a specific data split using BATCH estimator (legacy single-subject version)"""
    print(f"\n=== Computing temporal RUS for {split_name} split ===")

    # Determine unique activities and create mapping
    unique_activities = sorted([act for act in split_data['activity_id'].unique() if act != 0])
    activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}

    # Remap activity IDs
    split_data['activity_label'] = split_data['activity_id'].map(activity_map)
    split_data.dropna(subset=['activity_label'], inplace=True)
    split_data['activity_label'] = split_data['activity_label'].astype(int)

    Y = split_data['activity_label'].values
    
    if len(Y) <= args.max_lag:
        print(f"Error: Time series length ({len(Y)}) is not sufficient for max_lag ({args.max_lag}) for {split_name} split.")
        return None, None
    
    # Generate pairs of modalities
    modality_names = list(modality_sensors.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))
    print(f"Generated {len(modality_pairs)} pairs of modalities for analysis: {modality_pairs}")
    
    # Prepare data for each modality
    modality_data = {}
    for mod_name, sensors in modality_sensors.items():
        existing_sensors = [s for s in sensors if s in split_data.columns]
        if existing_sensors:
            modality_data[mod_name] = split_data[existing_sensors].values
            print(f"Modality '{mod_name}': {modality_data[mod_name].shape} (samples, features)")
        else:
            print(f"Warning: No existing sensors found for modality {mod_name}")
            continue
    
    all_pid_results = []
    
    for i, (mod1, mod2) in enumerate(modality_pairs):
        print(f"\n--- Analyzing Modality Pair {i+1}/{len(modality_pairs)}: {mod1} vs {mod2} ---")
        
        if mod1 not in modality_data or mod2 not in modality_data:
            print(f"Warning: Missing data for pair ({mod1}, {mod2}). Skipping.")
            continue
        
        X1 = modality_data[mod1]
        X2 = modality_data[mod2]
        
        if len(X1) != len(Y) or len(X2) != len(Y):
            print(f"Warning: Length mismatch for pair ({mod1}, {mod2}). Skipping.")
            continue
        
        print(f"Starting Multimodal Temporal PID analysis for {split_name} split...")
        print(f"X1 ({mod1}): {X1.shape} - {X1.shape[1]} features")
        print(f"X2 ({mod2}): {X2.shape} - {X2.shape[1]} features")
        print(f"Y: activity_id ({len(Y)} samples)")
        print(f"Max Lag: {args.max_lag}, Bins: {args.bins}")
        
        # Use BATCH method for RUS computation
        n_total_features = X1.shape[1] + X2.shape[1]
        print(f"Using BATCH method for total features: {n_total_features}")
        
        try:
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=args.max_lag, bins=args.bins,
                                            method='batch',
                                            batch_size=min(args.batch_size_rus, len(Y)//2),
                                            n_batches=args.n_batches,
                                            seed=args.seed,
                                            device=device,
                                            hidden_dim=args.hidden_dim,
                                            layers=args.layers,
                                            activation=args.activation,
                                            lr=args.lr_rus,
                                            embed_dim=args.embed_dim,
                                            discrim_epochs=args.discrim_epochs,
                                            ce_epochs=args.ce_epochs)
        except Exception as e:
            print(f"Error during PID analysis for pair ({mod1}, {mod2}): {e}")
            continue
        
        # Process results
        lags = pid_results.get('lag', range(args.max_lag + 1))
        lag_results = []
        
        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx]
                u2 = pid_results['unique_x2'][lag_idx]
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]
                
                if mi > 1e-9:
                    lag_results.append({
                        'lag': lag,
                        'R_value': r,
                        'U1_value': u1,
                        'U2_value': u2,
                        'S_value': s,
                        'MI_value': mi,
                        'R_norm': r / mi,
                        'U1_norm': u1 / mi,
                        'U2_norm': u2 / mi,
                        'S_norm': s / mi
                    })
            except (IndexError, KeyError) as e:
                print(f"Warning: Error processing lag {lag} for pair ({mod1}, {mod2}): {e}")
                continue
        
        if lag_results:
            # Calculate average metrics across all lags
            avg_metrics = {
                'R_value': np.mean([r['R_value'] for r in lag_results]),
                'U1_value': np.mean([r['U1_value'] for r in lag_results]),
                'U2_value': np.mean([r['U2_value'] for r in lag_results]),
                'S_value': np.mean([r['S_value'] for r in lag_results]),
                'MI_value': np.mean([r['MI_value'] for r in lag_results]),
                'R_norm': np.mean([r['R_norm'] for r in lag_results]),
                'U1_norm': np.mean([r['U1_norm'] for r in lag_results]),
                'U2_norm': np.mean([r['U2_norm'] for r in lag_results]),
                'S_norm': np.mean([r['S_norm'] for r in lag_results])
            }
            
            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': avg_metrics,
                'lag_results': lag_results,
                'modality1_features': modality_sensors[mod1],
                'modality2_features': modality_sensors[mod2],
                'n_features_mod1': len(modality_sensors[mod1]),
                'n_features_mod2': len(modality_sensors[mod2])
            })
    
    print(f"\nAnalysis complete for {split_name} split: {len(all_pid_results)} modality pairs processed.")
    
    return all_pid_results, activity_map


def save_rus_results(rus_results, args, split_name):
    """Save RUS results for a specific split"""
    if rus_results:
        if args.use_all_subjects:
            output_filename = f'pamap_all_subjects_{split_name}_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        else:
            output_filename = f'pamap_subject{args.subject_id}_{split_name}_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        output_path = os.path.join(args.output_dir, output_filename)
        print(f"Saving {len(rus_results)} RUS results for {split_name} split to {output_path}...")
        np.save(output_path, rus_results, allow_pickle=True)
        print(f"RUS results for {split_name} split saved successfully.")
        return output_path
    else:
        print(f"No RUS results to save for {split_name} split.")
        return None


class MultimodalPamapDatasetWithSplit(MultimodalPamapDataset):
    """Extended dataset class that works with pre-split data"""
    
    def __init__(self, split_data, rus_data, modality_sensors, seq_len, step, activity_map):
        """
        Initialize dataset with pre-split data
        
        Args:
            split_data: Pre-split DataFrame for this split (train/val/test)
            rus_data: RUS data computed for this split
            modality_sensors: Dictionary mapping modality names to sensor columns
            seq_len: Sequence length
            step: Step size for sliding window
            activity_map: Activity ID mapping
        """
        self.split_data = split_data
        self.rus_data = rus_data
        self.modality_sensors = modality_sensors
        self.modality_names = list(modality_sensors.keys())
        self.num_modalities = len(self.modality_names)
        self.seq_len = seq_len
        self.step = step
        self.activity_map = activity_map
        
        if self.split_data.empty:
            print("Warning: Split data is empty")
            self.windows = []
            self.labels = []
        else:
            self._create_windows()
    
    def _create_windows(self):
        """Creates sliding windows with separate features for each modality, handling multiple subjects properly."""
        self.windows = []
        self.labels = []

        # Check if we have multiple subjects
        if 'subject_id' in self.split_data.columns:
            # Process each subject separately to avoid cross-subject temporal dependencies
            subject_ids = self.split_data['subject_id'].unique()
            print(f"Processing {len(subject_ids)} subjects for window creation")

            for subject_id in subject_ids:
                subject_data = self.split_data[self.split_data['subject_id'] == subject_id]
                self._create_windows_for_subject(subject_data, subject_id)
        else:
            # Single subject or already processed data
            self._create_windows_for_subject(self.split_data, None)

        print(f"Created {len(self.windows)} multimodal windows for split data.")

    def _create_windows_for_subject(self, subject_data, subject_id):
        """Create windows for a single subject's data"""
        if len(subject_data) < self.seq_len:
            print(f"Warning: Subject {subject_id} has insufficient data ({len(subject_data)} < {self.seq_len})")
            return

        # Get data for each modality for this subject
        modality_data = {}
        for mod_name, sensors in self.modality_sensors.items():
            existing_sensors = [s for s in sensors if s in subject_data.columns]
            if existing_sensors:
                modality_data[mod_name] = subject_data[existing_sensors].values
            else:
                print(f"Warning: No sensors found for modality {mod_name} in subject {subject_id}")
                modality_data[mod_name] = np.zeros((len(subject_data), 1))

        label_values = subject_data['activity_label'].values
        total_samples = len(subject_data)

        for i in range(0, total_samples - self.seq_len + 1, self.step):
            window_data_by_modality = []

            for mod_name in self.modality_names:
                mod_window = modality_data[mod_name][i : i + self.seq_len]
                window_data_by_modality.append(torch.tensor(mod_window, dtype=torch.float32))

            window_labels = label_values[i : i + self.seq_len]
            unique_labels, counts = np.unique(window_labels, return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]

            self.windows.append(window_data_by_modality)
            self.labels.append(torch.tensor(most_frequent_label, dtype=torch.long))


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_config = {k: v for k, v in vars(args).items()}
        if args.use_all_subjects:
            run_name = f"full_exp_all_subjects_seq{args.seq_len}"
        else:
            run_name = f"full_exp_subj{args.subject_id}_seq{args.seq_len}"
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )
    
    print("=== FULL PAMAP2 EXPERIMENT: Temporal RUS + TRUS MoE ===")

    if args.use_all_subjects:
        # Parse subject lists
        train_subjects = [int(s.strip()) for s in args.train_subjects.split(',')]
        val_subjects = [int(s.strip()) for s in args.val_subjects.split(',')]
        test_subjects = [int(s.strip()) for s in args.test_subjects.split(',')]
        all_subjects = train_subjects + val_subjects + test_subjects

        print(f"Using all subjects approach:")
        print(f"  Train subjects: {train_subjects}")
        print(f"  Val subjects: {val_subjects}")
        print(f"  Test subjects: {test_subjects}")

        # Load and preprocess data for all subjects
        print("\n=== STEP 1: Loading and preprocessing data for all subjects ===")
        try:
            combined_df, sensor_columns = load_all_subjects_data(args.dataset_dir, all_subjects)

            if combined_df.empty:
                print("No data remaining after preprocessing. Exiting.")
                sys.exit(1)

            print(f"Combined preprocessed data shape: {combined_df.shape}")

        except Exception as e:
            print(f"Error loading/preprocessing data: {e}")
            sys.exit(1)

        # Create subject-based splits
        train_split_data, val_split_data, test_split_data = create_subject_based_splits(
            combined_df, train_subjects, val_subjects, test_subjects
        )

        # Create a mock splitter for modality_sensors
        from pamap_rus_multimodal import categorize_pamap_sensors
        modality_sensors = categorize_pamap_sensors(sensor_columns)

    else:
        print(f"Using single subject approach: Subject ID {args.subject_id}")
        print(f"Data splits: Train={args.train_split}, Val={args.val_split}, Test={args.test_split}")

        # Load and preprocess data
        print("\n=== STEP 1: Loading and preprocessing data ===")
        try:
            df = load_pamap_data(args.subject_id, args.dataset_dir)
            df_processed, sensor_columns = preprocess_pamap_data(df)

            if df_processed.empty:
                print("No data remaining after preprocessing. Exiting.")
                sys.exit(1)

            print(f"Preprocessed data shape: {df_processed.shape}")

        except Exception as e:
            print(f"Error loading/preprocessing data: {e}")
            sys.exit(1)

        # Create data splitter
        splitter = MultimodalDataSplitter(df_processed, sensor_columns, args)
        modality_sensors = splitter.modality_sensors
    
    print("\n=== STEP 2: Computing temporal RUS for each split ===")

    # Compute RUS for each split
    splits = ['train', 'val', 'test']
    rus_results_all = {}
    activity_maps_all = {}

    if args.use_all_subjects:
        # Use multi-subject approach
        split_data_dict = {
            'train': train_split_data,
            'val': val_split_data,
            'test': test_split_data
        }
        subject_ids_dict = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

        for split_name in splits:
            split_data = split_data_dict[split_name]
            subject_ids = subject_ids_dict[split_name]

            rus_results, activity_map = compute_temporal_rus_for_split_multi_subject(
                split_data, modality_sensors, args, split_name, device, subject_ids
            )

            if rus_results is not None:
                rus_results_all[split_name] = rus_results
                activity_maps_all[split_name] = activity_map

                # Save RUS results
                save_rus_results(rus_results, args, split_name)
            else:
                print(f"Failed to compute RUS for {split_name} split")
                sys.exit(1)

    else:
        # Use single subject approach
        for split_name in splits:
            split_data, split_indices = splitter.get_split_data(split_name)

            rus_results, activity_map = compute_temporal_rus_for_split(
                split_data, splitter.modality_sensors, args, split_name, device
            )

            if rus_results is not None:
                rus_results_all[split_name] = rus_results
                activity_maps_all[split_name] = activity_map

                # Save RUS results
                save_rus_results(rus_results, args, split_name)
            else:
                print(f"Failed to compute RUS for {split_name} split")
                sys.exit(1)
    
    print("\n=== STEP 3: Setting up TRUS MoE training ===")

    # Load RUS data for model training
    if args.use_all_subjects:
        train_rus_filename = f'pamap_all_subjects_train_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        val_rus_filename = f'pamap_all_subjects_val_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        test_rus_filename = f'pamap_all_subjects_test_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
    else:
        train_rus_filename = f'pamap_subject{args.subject_id}_train_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        val_rus_filename = f'pamap_subject{args.subject_id}_val_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        test_rus_filename = f'pamap_subject{args.subject_id}_test_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'

    train_rus_data = load_multimodal_rus_data(
        os.path.join(args.output_dir, train_rus_filename),
        modality_sensors, args.seq_len
    )

    val_rus_data = load_multimodal_rus_data(
        os.path.join(args.output_dir, val_rus_filename),
        modality_sensors, args.seq_len
    )

    test_rus_data = load_multimodal_rus_data(
        os.path.join(args.output_dir, test_rus_filename),
        modality_sensors, args.seq_len
    )

    # Get split data
    if args.use_all_subjects:
        # Data already prepared
        pass
    else:
        # Create datasets for each split
        train_split_data, _ = splitter.get_split_data('train')
        val_split_data, _ = splitter.get_split_data('val')
        test_split_data, _ = splitter.get_split_data('test')
    
    # Use train activity map for consistency
    activity_map = activity_maps_all['train']
    num_classes = len(activity_map)
    
    # Remap activities in all splits using the train activity map
    for split_name, split_data in [('train', train_split_data), ('val', val_split_data), ('test', test_split_data)]:
        split_data['activity_label'] = split_data['activity_id'].map(activity_map)
        split_data.dropna(subset=['activity_label'], inplace=True)
        split_data['activity_label'] = split_data['activity_label'].astype(int)
    
    train_dataset = MultimodalPamapDatasetWithSplit(
        train_split_data, train_rus_data, modality_sensors,
        args.seq_len, args.window_step, activity_map
    )

    val_dataset = MultimodalPamapDatasetWithSplit(
        val_split_data, val_rus_data, modality_sensors,
        args.seq_len, args.window_step, activity_map
    )

    test_dataset = MultimodalPamapDatasetWithSplit(
        test_split_data, test_rus_data, modality_sensors,
        args.seq_len, args.window_step, activity_map
    )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty.")
        sys.exit(1)
    
    # Create data loaders
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
    ) if len(val_dataset) > 0 else []
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    ) if len(test_dataset) > 0 else []
    
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
            'use_cnn': args.use_cnn_encoders and mod_name != 'heart_rate',
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
    
    print("Initializing multimodal TRUS-MoE model...")
    
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
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    task_criterion = nn.CrossEntropyLoss()
    
    print("\n=== STEP 4: Training TRUS MoE model ===")
    
    best_val_accuracy = -1.0
    best_epoch = -1
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch_multimodal(
            model, train_loader, optimizer, task_criterion, device, args, epoch
        )
        
        if len(val_loader) > 0:
            val_loss, val_acc = validate_epoch_multimodal(
                model, val_loader, task_criterion, device, args, epoch
            )
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                
                if args.use_all_subjects:
                    save_path = os.path.join(args.output_dir, f'best_full_exp_model_all_subjects.pth')
                else:
                    save_path = os.path.join(args.output_dir, f'best_full_exp_model_subj{args.subject_id}.pth')
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                    'args': args,
                    'modality_configs': modality_configs,
                    'modality_names': modality_names,
                    'activity_map': activity_map,
                }, save_path)
                
                print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%. Model saved.")
                
                if args.use_wandb:
                    wandb.log({"best_val_accuracy": best_val_accuracy, "best_epoch": epoch + 1})
        
        if scheduler is not None:
            scheduler.step()
    
    print("\n=== STEP 5: Final evaluation on test set ===")
    
    if len(test_loader) > 0 and best_epoch != -1:
        # Load best model for testing
        if args.use_all_subjects:
            best_model_path = os.path.join(args.output_dir, f'best_full_exp_model_all_subjects.pth')
        else:
            best_model_path = os.path.join(args.output_dir, f'best_full_exp_model_subj{args.subject_id}.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loss, test_acc = validate_epoch_multimodal(
                model, test_loader, task_criterion, device, args, args.epochs  # Use epochs as dummy epoch
            )
            
            print(f"Final Test Accuracy: {test_acc:.2f}%")
            
            if args.use_wandb:
                wandb.log({"test_accuracy": test_acc})
        else:
            print("Best model not found for testing.")
    else:
        print("Test evaluation skipped (no test data or no best model).")
    
    print("\n=== EXPERIMENT COMPLETED ===")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")
    
    # Finish wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()