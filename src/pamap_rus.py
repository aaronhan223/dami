import pandas as pd
import numpy as np
import math
import os
import sys
import argparse
import torch
import torch.nn as nn
from temporal_pid import multi_lag_analysis, plot_multi_lag_results
import matplotlib.pyplot as plt
import itertools
import pdb

# Define constants
DATASET_DIR = "/cis/home/xhan56/pamap/PAMAP2_Dataset/Protocol"
OUTPUT_DIR = "../results/pamap"
SUBJECT_ID = 1
MAX_LAG = 10
BINS = 8
DOMINANCE_THRESHOLD = 0.3 # Threshold for a PID term to be considered dominant

def get_pamap_column_names():
    """Returns the standard column names for PAMAP2 dataset files."""
    columns = ['timestamp', 'activity_id', 'heart_rate']
    imu_locs = ['hand', 'chest', 'ankle']
    imu_sensors = ['temp', 'acc16g_x', 'acc16g_y', 'acc16g_z',
                   'acc6g_x', 'acc6g_y', 'acc6g_z',
                   'gyro_x', 'gyro_y', 'gyro_z',
                   'mag_x', 'mag_y', 'mag_z',
                   'orient_w', 'orient_x', 'orient_y', 'orient_z']

    for loc in imu_locs:
        for sensor in imu_sensors:
            col_name = f"{sensor}_{loc}"
            columns.append(col_name)
    return columns

def load_pamap_data(subject_id, data_dir):
    """Loads data for a specific subject from the PAMAP2 dataset."""
    file_path = os.path.join(data_dir, f"subject10{subject_id}.dat")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for subject {subject_id} at {file_path}")

    print(f"Loading data for subject {subject_id} from {file_path}...")
    df = pd.read_csv(file_path, sep='\s+', header=None, names=get_pamap_column_names())
    print(f"Loaded data shape: {df.shape}")
    return df

def preprocess_pamap_data(df):
    """Preprocesses the loaded PAMAP2 data, keeping all sensor columns."""
    print("Preprocessing data...")
    essential_cols = ['timestamp', 'activity_id']
    sensor_cols = [col for col in df.columns if col not in essential_cols and 'orient' not in col]
    if 'heart_rate' not in df.columns:
        sensor_cols.insert(0, 'heart_rate')
    relevant_cols = essential_cols + sensor_cols
    df_processed = df[relevant_cols].copy()

    print(f"NaN counts before interpolation: {df_processed.isnull().sum()[df_processed.isnull().sum() > 0]}")
    df_processed = df_processed.interpolate(method='linear', limit_direction='both')

    if df_processed.isnull().sum().sum() > 0:
        print("Warning: NaNs still present after interpolation. Dropping rows with NaNs.")
        df_processed.dropna(inplace=True)

    df_processed = df_processed[df_processed['activity_id'] != 0]

    df_processed['activity_id'] = df_processed['activity_id'].astype(int)
    for col in sensor_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(float)

    print(f"Preprocessing complete. Data shape: {df_processed.shape}")
    print(f"Unique activities remaining: {df_processed['activity_id'].unique()}")
    print(f"Available sensor columns for analysis: {sensor_cols}")
    return df_processed, sensor_cols

def main():
    """Main function to load, preprocess, and analyze PAMAP2 data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = load_pamap_data(SUBJECT_ID, DATASET_DIR)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    df_processed, sensor_columns = preprocess_pamap_data(df)

    if df_processed.empty:
        print("No data remaining after preprocessing. Exiting.")
        sys.exit(1)

    Y = df_processed['activity_id'].values

    if len(Y) <= MAX_LAG:
        print(f"Error: Time series length ({len(Y)}) is not sufficient for max_lag ({MAX_LAG}). Aborting analysis.")
        sys.exit(1)

    sensor_pairs = list(itertools.combinations(sensor_columns, 2))
    print(f"Generated {len(sensor_pairs)} pairs of sensor variables for analysis.")

    dominant_pid_results = [] # List to store results where a term is dominant

    for i, (col1, col2) in enumerate(sensor_pairs):
        print(f"\n--- Analyzing Pair {i+1}/{len(sensor_pairs)}: {col1} vs {col2} --- ")

        X1 = df_processed[col1].values
        X2 = df_processed[col2].values
        if len(X1) != len(Y) or len(X2) != len(Y):
            print(f"Warning: Length mismatch for pair ({col1}, {col2}). Skipping.")
            print(f"Len X1: {len(X1)}, Len X2: {len(X2)}, Len Y: {len(Y)}")
            continue

        print(f"Starting Temporal PID analysis for Subject {SUBJECT_ID}...")
        print(f"X1: {col1} ({len(X1)} samples)")
        print(f"X2: {col2} ({len(X2)} samples)")
        print(f"Y: activity_id ({len(Y)} samples)")
        print(f"Max Lag: {MAX_LAG}, Bins: {BINS}")

        try:
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=MAX_LAG, bins=BINS)
        except Exception as e:
            print(f"Error during PID analysis for pair ({col1}, {col2}): {e}")
            continue

        # --- Check for dominance and collect results ---
        lags = pid_results.get('lag', range(MAX_LAG + 1)) # Assuming pid_results has 'lag' key or default to range
        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx] # Unique info from X1 (col1)
                u2 = pid_results['unique_x2'][lag_idx] # Unique info from X2 (col2)
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                if mi > 1e-9: # Avoid division by zero or near-zero MI
                    dominant_term = None
                    if r / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'R'
                    elif u1 / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'U1'
                    elif u2 / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'U2'
                    elif s / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'S'

                    if dominant_term:
                        dominant_pid_results.append({
                            'feature_pair': (col1, col2),
                            'lag': lag,
                            'dominant_term': dominant_term,
                            'R_value': r,
                            'U1_value': u1,
                            'U2_value': u2,
                            'S_value': s,
                            'MI_value': mi
                        })
            except IndexError:
                 print(f"Warning: Index out of bounds for lag {lag} (index {lag_idx}) for pair ({col1}, {col2}). Skipping lag.")
                 continue
            except KeyError as e:
                 print(f"Warning: Missing key {e} in pid_results for pair ({col1}, {col2}). Skipping dominance check.")
                 break # Stop checking lags for this pair if keys are missing

        # --- Commented out plotting ---
        # sanitized_col1 = col1.replace('_', '-').replace('.', '')
        # sanitized_col2 = col2.replace('_', '-').replace('.', '')
        # plot_filename = f'pamap_subject{SUBJECT_ID}_pid_{sanitized_col1}_vs_{sanitized_col2}_lag{MAX_LAG}_bins{BINS}.png'
        # plot_save_path = os.path.join(OUTPUT_DIR, plot_filename)
        # print(f"Plotting results to {plot_save_path}...")
        # try:
        #     plot_multi_lag_results(pid_results, title=f'PID: {col1} vs {col2} (Subject {SUBJECT_ID})', save_path=plot_save_path)
        # except Exception as e:
        #     print(f"Error plotting results for pair ({col1}, {col2}): {e}")
        # finally:
        #     plt.close()

    print(f"\nAnalysis complete for all {len(sensor_pairs)} pairs.")

    # --- Save dominant PID results ---
    if dominant_pid_results:
        output_filename = f'pamap_subject{SUBJECT_ID}_lag{MAX_LAG}_bins{BINS}_thresh{DOMINANCE_THRESHOLD:.1f}.npy'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        print(f"Saving {len(dominant_pid_results)} dominant PID results to {output_path}...")
        np.save(output_path, dominant_pid_results, allow_pickle=True) # Need allow_pickle=True for list of dicts
        print("Saving complete.")

        # --- Print summary of dominant terms ---
        dominance_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        for result in dominant_pid_results:
            term = result.get('dominant_term')
            if term in dominance_counts:
                dominance_counts[term] += 1

        print("\n--- Dominance Summary ---")
        print(f"Total dominant instances found: {len(dominant_pid_results)}")
        for term, count in dominance_counts.items():
            print(f"  {term} dominant: {count} times")
        print("-------------------------")

    else:
        print("No dominant PID terms found with the current threshold.")

if __name__ == "__main__":
    main()

