#!/usr/bin/env python3
"""
Full Opportunity Experiment: Temporal RUS Computation + Multimodal TRUS MoE Training

This script performs:
1. Temporal RUS computation using BATCH estimator
2. Multimodal TRUS MoE training

Key requirements:
- Train/val/test split: subjects 1-3 runs 1-4 for training, subjects 1-3 run 5 for validation, subject 4 all runs for testing
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
    from trus_moe_multimodal import MultimodalTRUSMoEModel
    from trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
    from plots.plot_expert_activation import analyze_expert_activations
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)


DEFAULT_DATASET_DIR = "/cis/home/xhan56/OpportunityUCIDataset/dataset"
DEFAULT_OUTPUT_DIR = "../results/opportunity_full_exp"


def get_opportunity_column_names():
    """Get column names for Opportunity dataset based on the column_names.txt"""
    column_names = [
        'timestamp',
        # Body-worn accelerometers (2-37)
        'acc_RKN^_x', 'acc_RKN^_y', 'acc_RKN^_z',
        'acc_HIP_x', 'acc_HIP_y', 'acc_HIP_z',
        'acc_LUA^_x', 'acc_LUA^_y', 'acc_LUA^_z',
        'acc_RUA__x', 'acc_RUA__y', 'acc_RUA__z',
        'acc_LH_x', 'acc_LH_y', 'acc_LH_z',
        'acc_BACK_x', 'acc_BACK_y', 'acc_BACK_z',
        'acc_RKN__x', 'acc_RKN__y', 'acc_RKN__z',
        'acc_RWR_x', 'acc_RWR_y', 'acc_RWR_z',
        'acc_RUA^_x', 'acc_RUA^_y', 'acc_RUA^_z',
        'acc_LUA__x', 'acc_LUA__y', 'acc_LUA__z',
        'acc_LWR_x', 'acc_LWR_y', 'acc_LWR_z',
        'acc_RH_x', 'acc_RH_y', 'acc_RH_z',

        # IMU sensors (38-102)
        'imu_BACK_acc_x', 'imu_BACK_acc_y', 'imu_BACK_acc_z',
        'imu_BACK_gyro_x', 'imu_BACK_gyro_y', 'imu_BACK_gyro_z',
        'imu_BACK_mag_x', 'imu_BACK_mag_y', 'imu_BACK_mag_z',
        'imu_BACK_quat_1', 'imu_BACK_quat_2', 'imu_BACK_quat_3', 'imu_BACK_quat_4',

        'imu_RUA_acc_x', 'imu_RUA_acc_y', 'imu_RUA_acc_z',
        'imu_RUA_gyro_x', 'imu_RUA_gyro_y', 'imu_RUA_gyro_z',
        'imu_RUA_mag_x', 'imu_RUA_mag_y', 'imu_RUA_mag_z',
        'imu_RUA_quat_1', 'imu_RUA_quat_2', 'imu_RUA_quat_3', 'imu_RUA_quat_4',

        'imu_RLA_acc_x', 'imu_RLA_acc_y', 'imu_RLA_acc_z',
        'imu_RLA_gyro_x', 'imu_RLA_gyro_y', 'imu_RLA_gyro_z',
        'imu_RLA_mag_x', 'imu_RLA_mag_y', 'imu_RLA_mag_z',
        'imu_RLA_quat_1', 'imu_RLA_quat_2', 'imu_RLA_quat_3', 'imu_RLA_quat_4',

        'imu_LUA_acc_x', 'imu_LUA_acc_y', 'imu_LUA_acc_z',
        'imu_LUA_gyro_x', 'imu_LUA_gyro_y', 'imu_LUA_gyro_z',
        'imu_LUA_mag_x', 'imu_LUA_mag_y', 'imu_LUA_mag_z',
        'imu_LUA_quat_1', 'imu_LUA_quat_2', 'imu_LUA_quat_3', 'imu_LUA_quat_4',

        'imu_LLA_acc_x', 'imu_LLA_acc_y', 'imu_LLA_acc_z',
        'imu_LLA_gyro_x', 'imu_LLA_gyro_y', 'imu_LLA_gyro_z',
        'imu_LLA_mag_x', 'imu_LLA_mag_y', 'imu_LLA_mag_z',
        'imu_LLA_quat_1', 'imu_LLA_quat_2', 'imu_LLA_quat_3', 'imu_LLA_quat_4',

        # IMU shoe sensors (103-134)
        'imu_L_SHOE_eu_x', 'imu_L_SHOE_eu_y', 'imu_L_SHOE_eu_z',
        'imu_L_SHOE_nav_acc_x', 'imu_L_SHOE_nav_acc_y', 'imu_L_SHOE_nav_acc_z',
        'imu_L_SHOE_body_acc_x', 'imu_L_SHOE_body_acc_y', 'imu_L_SHOE_body_acc_z',
        'imu_L_SHOE_angvel_body_x', 'imu_L_SHOE_angvel_body_y', 'imu_L_SHOE_angvel_body_z',
        'imu_L_SHOE_angvel_nav_x', 'imu_L_SHOE_angvel_nav_y', 'imu_L_SHOE_angvel_nav_z',
        'imu_L_SHOE_compass',

        'imu_R_SHOE_eu_x', 'imu_R_SHOE_eu_y', 'imu_R_SHOE_eu_z',
        'imu_R_SHOE_nav_acc_x', 'imu_R_SHOE_nav_acc_y', 'imu_R_SHOE_nav_acc_z',
        'imu_R_SHOE_body_acc_x', 'imu_R_SHOE_body_acc_y', 'imu_R_SHOE_body_acc_z',
        'imu_R_SHOE_angvel_body_x', 'imu_R_SHOE_angvel_body_y', 'imu_R_SHOE_angvel_body_z',
        'imu_R_SHOE_angvel_nav_x', 'imu_R_SHOE_angvel_nav_y', 'imu_R_SHOE_angvel_nav_z',
        'imu_R_SHOE_compass',

        # Object accelerometers (135-194)
        'acc_CUP_x', 'acc_CUP_y', 'acc_CUP_z', 'acc_CUP_gyro_x', 'acc_CUP_gyro_y',
        'acc_SALAMI_x', 'acc_SALAMI_y', 'acc_SALAMI_z', 'acc_SALAMI_gyro_x', 'acc_SALAMI_gyro_y',
        'acc_WATER_x', 'acc_WATER_y', 'acc_WATER_z', 'acc_WATER_gyro_x', 'acc_WATER_gyro_y',
        'acc_CHEESE_x', 'acc_CHEESE_y', 'acc_CHEESE_z', 'acc_CHEESE_gyro_x', 'acc_CHEESE_gyro_y',
        'acc_BREAD_x', 'acc_BREAD_y', 'acc_BREAD_z', 'acc_BREAD_gyro_x', 'acc_BREAD_gyro_y',
        'acc_KNIFE1_x', 'acc_KNIFE1_y', 'acc_KNIFE1_z', 'acc_KNIFE1_gyro_x', 'acc_KNIFE1_gyro_y',
        'acc_MILK_x', 'acc_MILK_y', 'acc_MILK_z', 'acc_MILK_gyro_x', 'acc_MILK_gyro_y',
        'acc_SPOON_x', 'acc_SPOON_y', 'acc_SPOON_z', 'acc_SPOON_gyro_x', 'acc_SPOON_gyro_y',
        'acc_SUGAR_x', 'acc_SUGAR_y', 'acc_SUGAR_z', 'acc_SUGAR_gyro_x', 'acc_SUGAR_gyro_y',
        'acc_KNIFE2_x', 'acc_KNIFE2_y', 'acc_KNIFE2_z', 'acc_KNIFE2_gyro_x', 'acc_KNIFE2_gyro_y',
        'acc_PLATE_x', 'acc_PLATE_y', 'acc_PLATE_z', 'acc_PLATE_gyro_x', 'acc_PLATE_gyro_y',
        'acc_GLASS_x', 'acc_GLASS_y', 'acc_GLASS_z', 'acc_GLASS_gyro_x', 'acc_GLASS_gyro_y',

        # Reed switches (195-207)
        'reed_DISHWASHER_S1', 'reed_FRIDGE_S3', 'reed_FRIDGE_S2', 'reed_FRIDGE_S1',
        'reed_MIDDLEDRAWER_S1', 'reed_MIDDLEDRAWER_S2', 'reed_MIDDLEDRAWER_S3',
        'reed_LOWERDRAWER_S3', 'reed_LOWERDRAWER_S2', 'reed_UPPERDRAWER',
        'reed_DISHWASHER_S3', 'reed_LOWERDRAWER_S1', 'reed_DISHWASHER_S2',

        # Environmental accelerometers (208-231)
        'acc_DOOR1_x', 'acc_DOOR1_y', 'acc_DOOR1_z',
        'acc_LAZYCHAIR_x', 'acc_LAZYCHAIR_y', 'acc_LAZYCHAIR_z',
        'acc_DOOR2_x', 'acc_DOOR2_y', 'acc_DOOR2_z',
        'acc_DISHWASHER_x', 'acc_DISHWASHER_y', 'acc_DISHWASHER_z',
        'acc_UPPERDRAWER_x', 'acc_UPPERDRAWER_y', 'acc_UPPERDRAWER_z',
        'acc_LOWERDRAWER_x', 'acc_LOWERDRAWER_y', 'acc_LOWERDRAWER_z',
        'acc_MIDDLEDRAWER_x', 'acc_MIDDLEDRAWER_y', 'acc_MIDDLEDRAWER_z',
        'acc_FRIDGE_x', 'acc_FRIDGE_y', 'acc_FRIDGE_z',

        # Location tags (232-243)
        'loc_TAG1_x', 'loc_TAG1_y', 'loc_TAG1_z',
        'loc_TAG2_x', 'loc_TAG2_y', 'loc_TAG2_z',
        'loc_TAG3_x', 'loc_TAG3_y', 'loc_TAG3_z',
        'loc_TAG4_x', 'loc_TAG4_y', 'loc_TAG4_z',

        # Labels (244-250)
        'locomotion', 'hl_activity', 'll_left_arm', 'll_left_arm_object',
        'll_right_arm', 'll_right_arm_object', 'ml_both_arms'
    ]
    return column_names


def categorize_opportunity_sensors(sensor_columns: List[str]) -> Dict[str, List[str]]:
    """
    Categorizes Opportunity sensor columns by modality type and body location.

    Returns:
        Dictionary mapping modality names to lists of sensor columns
    """
    modality_sensors = {
        'torso': [],           # Back, hip sensors
        'arms': [],            # Upper arm, wrist, hand sensors
        'legs': [],            # Knee, ankle, foot sensors
        'shoes': [],           # Shoe IMU sensors
        'objects': [],         # Object-mounted sensors (cups, tools, etc.)
        'environment': [],     # Environmental sensors (doors, drawers, etc.)
        'location': []         # Location tracking sensors
    }

    for col in sensor_columns:
        col_lower = col.lower()

        # Torso sensors (back, hip)
        if any(x in col_lower for x in ['back', 'hip']):
            modality_sensors['torso'].append(col)

        # Arm sensors (upper arm, wrist, hand)
        elif any(x in col_lower for x in ['ua', 'wr', '_lh_', '_rh_', 'arm']):
            modality_sensors['arms'].append(col)

        # Leg sensors (knee, ankle)
        elif any(x in col_lower for x in ['kn', 'la', 'leg']):
            modality_sensors['legs'].append(col)

        # Shoe sensors
        elif 'shoe' in col_lower:
            modality_sensors['shoes'].append(col)

        # Object sensors (cup, salami, water, etc.)
        elif any(x in col_lower for x in ['cup', 'salami', 'water', 'cheese', 'bread',
                                         'knife', 'milk', 'spoon', 'sugar', 'plate', 'glass']):
            modality_sensors['objects'].append(col)

        # Environmental sensors (doors, drawers, dishwasher, fridge, etc.)
        elif any(x in col_lower for x in ['door', 'drawer', 'dishwasher', 'fridge',
                                         'lazychair', 'reed']):
            modality_sensors['environment'].append(col)

        # Location sensors
        elif any(x in col_lower for x in ['tag', 'loc']):
            modality_sensors['location'].append(col)

        else:
            print(f"Warning: Could not categorize sensor column: {col}")

    # Remove empty modalities
    modality_sensors = {k: v for k, v in modality_sensors.items() if v}

    return modality_sensors


def load_opportunity_data(subject_id: int, run: str, data_dir: str) -> pd.DataFrame:
    """Load Opportunity dataset for a specific subject and run"""
    filename = f"S{subject_id}-{run}.dat"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading {filename}...")

    # Load data without headers
    data = pd.read_csv(filepath, sep=' ', header=None, na_values='NaN')

    # Get column names
    column_names = get_opportunity_column_names()

    # Ensure we have the right number of columns
    if len(data.columns) != len(column_names):
        print(f"Warning: Expected {len(column_names)} columns, got {len(data.columns)}")
        # Truncate column names if necessary
        column_names = column_names[:len(data.columns)]

    data.columns = column_names

    print(f"Loaded data shape: {data.shape}")
    return data


def preprocess_opportunity_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Preprocess Opportunity data"""
    print("Preprocessing Opportunity data...")

    # Get all sensor columns (exclude timestamp and labels)
    label_cols = ['locomotion', 'hl_activity', 'll_left_arm', 'll_left_arm_object',
                  'll_right_arm', 'll_right_arm_object', 'ml_both_arms']
    sensor_columns = [col for col in df.columns if col not in ['timestamp'] + label_cols]

    # Convert timestamp to numeric if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # Convert sensor columns to numeric
    for col in sensor_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Use high-level activity as the primary label since locomotion is often 0
    if 'hl_activity' in df.columns:
        df['activity_id'] = df['hl_activity']
    elif 'locomotion' in df.columns:
        df['activity_id'] = df['locomotion']
    else:
        df['activity_id'] = 0

    # Convert activity_id to numeric
    df['activity_id'] = pd.to_numeric(df['activity_id'], errors='coerce')

    # Remove rows with missing activity labels
    df = df.dropna(subset=['activity_id'])
    df = df[df['activity_id'] != 0]  # Remove null activity (0)

    # Handle missing sensor values - forward fill then backward fill
    df[sensor_columns] = df[sensor_columns].ffill().bfill()

    # Remove columns that are still mostly NaN (more than 50% missing)
    valid_sensors = []
    for col in sensor_columns:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio < 0.5:  # Keep columns with less than 50% NaN
            valid_sensors.append(col)

    print(f"Keeping {len(valid_sensors)} out of {len(sensor_columns)} sensor columns (>50% valid data)")
    sensor_columns = valid_sensors

    # Remove rows that still have NaN values in the valid sensors
    df = df.dropna(subset=sensor_columns)

    print(f"After preprocessing: {df.shape}")
    print(f"Unique activities: {sorted(df['activity_id'].unique())}")

    return df, sensor_columns


def load_subjects_data(data_dir: str, subjects_runs: Dict[int, List[str]]) -> pd.DataFrame:
    """Load and combine data from multiple subjects and runs"""
    print(f"Loading data for subjects and runs: {subjects_runs}")

    all_data = []
    for subject_id, runs in subjects_runs.items():
        for run in runs:
            try:
                df = load_opportunity_data(subject_id, run, data_dir)
                df_processed, sensor_columns = preprocess_opportunity_data(df)

                if not df_processed.empty:
                    # Add subject_id and run columns to track data source
                    df_processed['subject_id'] = subject_id
                    df_processed['run'] = run
                    all_data.append(df_processed)
                    print(f"Subject {subject_id}, Run {run}: {df_processed.shape[0]} samples")
                else:
                    print(f"Warning: No data for Subject {subject_id}, Run {run} after preprocessing")
            except Exception as e:
                print(f"Error loading Subject {subject_id}, Run {run}: {e}")
                continue

    if not all_data:
        raise ValueError("No valid data loaded from any subject/run")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")

    return combined_df, sensor_columns


def create_opportunity_splits(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Create train/val/test splits as specified for Opportunity dataset"""

    # Define splits according to requirements
    train_subjects_runs = {
        1: ['ADL1', 'ADL2', 'ADL3', 'ADL4'],
        2: ['ADL1', 'ADL2', 'ADL3', 'ADL4'],
        3: ['ADL1', 'ADL2', 'ADL3', 'ADL4']
    }

    val_subjects_runs = {
        1: ['ADL5'],
        2: ['ADL5'],
        3: ['ADL5']
    }

    test_subjects_runs = {
        4: ['ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5']
    }

    # Load data for each split
    print("Loading training data...")
    train_data, sensor_columns = load_subjects_data(data_dir, train_subjects_runs)

    print("Loading validation data...")
    val_data, _ = load_subjects_data(data_dir, val_subjects_runs)

    print("Loading test data...")
    test_data, _ = load_subjects_data(data_dir, test_subjects_runs)

    print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data, sensor_columns


def prepare_lagged_data_per_subject(split_data, modality_sensors, max_lag):
    """Prepare lagged data by processing each subject individually to prevent cross-subject time mixing"""
    # Get all unique (subject_id, run) combinations from the split data
    if 'subject_id' in split_data.columns and 'run' in split_data.columns:
        split_sessions = split_data.groupby(['subject_id', 'run'])
    else:
        # Fallback: treat as single session
        split_sessions = [(None, split_data)]

    # Generate pairs of modalities
    modality_names = list(modality_sensors.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))

    # Prepare combined data for each modality pair
    combined_data = {}

    for mod1, mod2 in modality_pairs:
        combined_X1 = []
        combined_X2 = []
        combined_Y = []

        for session_key, session_data in split_sessions:
            if len(session_data) <= max_lag:
                if session_key:
                    print(f"Warning: Session {session_key} has insufficient data ({len(session_data)} <= {max_lag})")
                continue

            # Prepare modality data for this session
            session_mod1_sensors = [s for s in modality_sensors[mod1] if s in session_data.columns]
            session_mod2_sensors = [s for s in modality_sensors[mod2] if s in session_data.columns]

            if not session_mod1_sensors or not session_mod2_sensors:
                if session_key:
                    print(f"Warning: Missing sensors for session {session_key}")
                continue

            X1_session = session_data[session_mod1_sensors].values
            X2_session = session_data[session_mod2_sensors].values
            Y_session = session_data['activity_label'].values

            # Add session data to combined arrays
            combined_X1.append(X1_session)
            combined_X2.append(X2_session)
            combined_Y.append(Y_session)

        # Store combined data for this modality pair
        if combined_X1:
            combined_data[(mod1, mod2)] = {
                'X1': combined_X1,
                'X2': combined_X2,
                'Y': combined_Y
            }

    return combined_data


def multi_lag_analysis_multi_session(X1_list, X2_list, Y_list, max_lag, bins, **kwargs):
    """
    Modified multi_lag_analysis that handles multiple sessions by creating
    lagged datasets for each session individually, then concatenating
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
        # Prepare lagged data for each session
        combined_X1_lag = []
        combined_X2_lag = []
        combined_Y_lag = []

        for X1_sess, X2_sess, Y_sess in zip(X1_list, X2_list, Y_list):
            if lag > 0:
                if len(Y_sess) <= lag:
                    continue
                X1_lagged = X1_sess[:-lag]
                X2_lagged = X2_sess[:-lag]
                Y_lagged = Y_sess[lag:]
            else:
                X1_lagged = X1_sess
                X2_lagged = X2_sess
                Y_lagged = Y_sess

            if len(Y_lagged) > 0:
                combined_X1_lag.append(X1_lagged)
                combined_X2_lag.append(X2_lagged)
                combined_Y_lag.append(Y_lagged)

        if not combined_X1_lag:
            continue

        # Concatenate lagged data from all sessions
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
    """Compute temporal RUS for a specific data split using BATCH estimator"""
    print(f"\n=== Computing temporal RUS for {split_name} split ===")

    # Determine unique activities and create mapping
    unique_activities = sorted([act for act in split_data['activity_id'].unique() if act != 0])
    activity_map = {activity_id: i for i, activity_id in enumerate(unique_activities)}

    # Remap activity IDs
    split_data['activity_label'] = split_data['activity_id'].map(activity_map)
    split_data.dropna(subset=['activity_label'], inplace=True)
    split_data['activity_label'] = split_data['activity_label'].astype(int)

    # Prepare combined data for multi-session analysis
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
            # Use multi-session analysis
            pid_results = multi_lag_analysis_multi_session(
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


def load_multimodal_rus_data(rus_filepath: str, modality_sensors: Dict[str, List[str]],
                            seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Loads RUS data and computes modality-level RUS values.
    """
    if not os.path.exists(rus_filepath):
        raise FileNotFoundError(f"RUS data file not found: {rus_filepath}")

    print(f"Loading RUS data from: {rus_filepath}")
    all_pid_results = np.load(rus_filepath, allow_pickle=True)

    # Get modality names and create mapping to index
    modality_names = list(modality_sensors.keys())
    num_modalities = len(modality_names)
    modality_to_idx = {name: idx for idx, name in enumerate(modality_names)}

    T = seq_len
    # Initialize modality-level tensors
    U = torch.zeros(num_modalities, T, dtype=torch.float32)
    R = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)
    S = torch.zeros(num_modalities, num_modalities, T, dtype=torch.float32)

    # We'll keep track of processed modality pairs to avoid duplicates
    processed_pairs = set()

    for result in all_pid_results:
        # The feature_pair now contains two modality names
        mod1, mod2 = result['feature_pair']

        # Skip if either modality is not in our list
        if mod1 not in modality_to_idx or mod2 not in modality_to_idx:
            print(f"Warning: Skipping pair ({mod1}, {mod2}) because one or both modalities not found.")
            continue

        m1_idx = modality_to_idx[mod1]
        m2_idx = modality_to_idx[mod2]

        # Skip if same modality (shouldn't happen, but just in case)
        if m1_idx == m2_idx:
            continue

        # Create a key for the unordered pair
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

            # Get the R, S, and U values for this lag
            R_value = lag_data['R_value']
            S_value = lag_data['S_value']
            U1_value = lag_data['U1_value']
            U2_value = lag_data['U2_value']

            # Update R and S for the pair (symmetric)
            R[m1_idx, m2_idx, start_idx:end_idx] = torch.tensor(R_value)
            R[m2_idx, m1_idx, start_idx:end_idx] = torch.tensor(R_value)

            S[m1_idx, m2_idx, start_idx:end_idx] = torch.tensor(S_value)
            S[m2_idx, m1_idx, start_idx:end_idx] = torch.tensor(S_value)

            # Update U for each modality: take the max of the current segment and the new U value
            # For modality m1_idx
            current_segment_U1 = U[m1_idx, start_idx:end_idx]
            current_max_U1 = current_segment_U1.max().item()
            new_value_U1 = max(current_max_U1, U1_value)
            U[m1_idx, start_idx:end_idx] = torch.tensor(new_value_U1)

            # For modality m2_idx
            current_segment_U2 = U[m2_idx, start_idx:end_idx]
            current_max_U2 = current_segment_U2.max().item()
            new_value_U2 = max(current_max_U2, U2_value)
            U[m2_idx, start_idx:end_idx] = torch.tensor(new_value_U2)

    print(f"Modality-level RUS data computed. Shapes: U({U.shape}), R({R.shape}), S({S.shape})")
    print(f"  Average R value: {R.mean().item():.4f}")
    print(f"  Average S value: {S.mean().item():.4f}")
    print(f"  Average U value: {U.mean().item():.4f}")

    return {'U': U, 'R': R, 'S': S}


class MultimodalOpportunityDataset(Dataset):
    """
    PyTorch Dataset for multimodal Opportunity activity recognition.
    """
    def __init__(self, split_data, rus_data, modality_sensors, seq_len, step, activity_map):
        """
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
        """Creates sliding windows with separate features for each modality, handling multiple sessions properly."""
        self.windows = []
        self.labels = []

        # Check if we have multiple sessions (subject_id, run combinations)
        if 'subject_id' in self.split_data.columns and 'run' in self.split_data.columns:
            # Process each session separately to avoid cross-session temporal dependencies
            sessions = self.split_data.groupby(['subject_id', 'run'])
            print(f"Processing {len(sessions)} sessions for window creation")

            for (subject_id, run), session_data in sessions:
                self._create_windows_for_session(session_data, f"S{subject_id}-{run}")
        else:
            # Single session or already processed data
            self._create_windows_for_session(self.split_data, "single_session")

        print(f"Created {len(self.windows)} multimodal windows for split data.")

    def _create_windows_for_session(self, session_data, session_id):
        """Create windows for a single session's data"""
        if len(session_data) < self.seq_len:
            print(f"Warning: Session {session_id} has insufficient data ({len(session_data)} < {self.seq_len})")
            return

        # Get data for each modality for this session
        modality_data = {}
        for mod_name, sensors in self.modality_sensors.items():
            existing_sensors = [s for s in sensors if s in session_data.columns]
            if existing_sensors:
                modality_data[mod_name] = session_data[existing_sensors].values
            else:
                print(f"Warning: No sensors found for modality {mod_name} in session {session_id}")
                modality_data[mod_name] = np.zeros((len(session_data), 1))

        label_values = session_data['activity_label'].values
        total_samples = len(session_data)

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

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.rus_data, self.labels[idx]


def collate_multimodal(batch):
    """Custom collate function for multimodal data."""
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


def train_epoch_multimodal(model: MultimodalTRUSMoEModel,
                          dataloader: DataLoader,
                          optimizer: optim.Optimizer,
                          task_criterion: nn.Module,
                          device: torch.device,
                          args: argparse.Namespace,
                          current_epoch: int):
    """Runs one training epoch for multimodal model."""
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
        # Move data to device
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

    # Log metrics to wandb
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


def validate_epoch_multimodal(model: MultimodalTRUSMoEModel,
                             dataloader: DataLoader,
                             task_criterion: nn.Module,
                             device: torch.device,
                             args: argparse.Namespace,
                             current_epoch: int):
    """Runs one validation epoch for multimodal model."""
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

    # Log validation metrics to wandb
    if args.use_wandb:
        wandb.log({
            "val/task_loss": avg_task_loss,
            "val/accuracy": accuracy,
            "epoch": current_epoch + 1
        })

    print(f"Epoch {current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy


def save_rus_results(rus_results, args, split_name):
    """Save RUS results for a specific split"""
    if rus_results:
        output_filename = f'opportunity_{split_name}_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
        output_path = os.path.join(args.output_dir, output_filename)
        print(f"Saving {len(rus_results)} RUS results for {split_name} split to {output_path}...")
        np.save(output_path, rus_results, allow_pickle=True)
        print(f"RUS results for {split_name} split saved successfully.")
        return output_path
    else:
        print(f"No RUS results to save for {split_name} split.")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Full Opportunity Experiment: RUS Computation + TRUS MoE Training')

    # Data args
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                        help='Directory containing Opportunity dataset files')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save analysis results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # RUS computation args (BATCH estimator)
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Maximum lag for temporal PID analysis')
    parser.add_argument('--bins', type=int, default=4,
                        help='Number of bins for discretization')

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
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    # Wandb args
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='opportunity-full-exp',
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
        run_name = f"opportunity_full_exp_seq{args.seq_len}"
        if args.wandb_run_name:
            run_name = args.wandb_run_name

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )

    print("=== FULL OPPORTUNITY EXPERIMENT: Temporal RUS + TRUS MoE ===")

    # Step 1: Load and prepare data splits
    print("\n=== STEP 1: Loading and preparing data splits ===")
    try:
        train_data, val_data, test_data, sensor_columns = create_opportunity_splits(args.dataset_dir)

        if train_data.empty or val_data.empty or test_data.empty:
            print("Error: One or more data splits are empty. Exiting.")
            sys.exit(1)

        print(f"Data loaded successfully.")
        print(f"Train: {len(train_data)} samples")
        print(f"Val: {len(val_data)} samples")
        print(f"Test: {len(test_data)} samples")

    except Exception as e:
        print(f"Error loading/preparing data: {e}")
        sys.exit(1)

    # Categorize sensors by modality
    modality_sensors = categorize_opportunity_sensors(sensor_columns)

    if not modality_sensors:
        print("Error: No sensors could be categorized into modalities.")
        sys.exit(1)

    print(f"Found {len(modality_sensors)} modalities:")
    for mod_name, sensors in modality_sensors.items():
        print(f"  {mod_name}: {len(sensors)} sensors")

    # Step 2: Compute temporal RUS for each split
    print("\n=== STEP 2: Computing temporal RUS for each split ===")

    splits = ['train', 'val', 'test']
    split_data_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    rus_results_all = {}
    activity_maps_all = {}

    for split_name in splits:
        split_data = split_data_dict[split_name]

        rus_results, activity_map = compute_temporal_rus_for_split(
            split_data, modality_sensors, args, split_name, device
        )

        if rus_results is not None:
            rus_results_all[split_name] = rus_results
            activity_maps_all[split_name] = activity_map

            # Save RUS results
            save_rus_results(rus_results, args, split_name)
        else:
            print(f"Failed to compute RUS for {split_name} split")
            sys.exit(1)

    # Step 3: Set up TRUS MoE training
    print("\n=== STEP 3: Setting up TRUS MoE training ===")

    # Load RUS data for model training
    train_rus_filename = f'opportunity_train_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
    val_rus_filename = f'opportunity_val_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'
    test_rus_filename = f'opportunity_test_multimodal_all_lag{args.max_lag}_bins{args.bins}.npy'

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

    # Use train activity map for consistency
    activity_map = activity_maps_all['train']
    num_classes = len(activity_map)

    # Remap activities in all splits using the train activity map
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_data['activity_label'] = split_data['activity_id'].map(activity_map)
        split_data.dropna(subset=['activity_label'], inplace=True)
        split_data['activity_label'] = split_data['activity_label'].astype(int)

    # Create datasets
    train_dataset = MultimodalOpportunityDataset(
        train_data, train_rus_data, modality_sensors,
        args.seq_len, args.window_step, activity_map
    )

    val_dataset = MultimodalOpportunityDataset(
        val_data, val_rus_data, modality_sensors,
        args.seq_len, args.window_step, activity_map
    )

    test_dataset = MultimodalOpportunityDataset(
        test_data, test_rus_data, modality_sensors,
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

    # Step 4: Training TRUS MoE model
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

                save_path = os.path.join(args.output_dir, f'best_opportunity_model.pth')

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

    # Step 5: Final evaluation on test set
    print("\n=== STEP 5: Final evaluation on test set ===")

    if len(test_loader) > 0 and best_epoch != -1:
        # Load best model for testing
        best_model_path = os.path.join(args.output_dir, f'best_opportunity_model.pth')
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