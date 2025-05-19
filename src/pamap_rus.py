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
SUBJECT_ID = 9
MAX_LAG = 10
BINS = 8
DOMINANCE_THRESHOLD = 0.4 # Threshold for a PID term to be considered dominant

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
    DOMINANCE_PERCENTAGE = 0.9  # 90% threshold for dominance across time lags

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

        # --- Analyze dominance across all lags as a unit ---
        lags = pid_results.get('lag', range(MAX_LAG + 1))
        dominant_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        total_valid_lags = 0
        
        lag_results = []  # Store results for all lags for this pair
        
        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx]
                u2 = pid_results['unique_x2'][lag_idx]
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                if mi > 1e-9:  # Avoid division by zero or near-zero MI
                    total_valid_lags += 1
                    
                    # Get normalized values for each term
                    r_norm = r / mi
                    u1_norm = u1 / mi
                    u2_norm = u2 / mi
                    s_norm = s / mi
                    
                    # Find the term with the highest value
                    norm_values = {
                        'R': r_norm,
                        'U1': u1_norm,
                        'U2': u2_norm,
                        'S': s_norm
                    }
                    max_term = max(norm_values, key=norm_values.get)
                    max_value = norm_values[max_term]
                    
                    # If highest and above threshold, count it
                    if max_value > DOMINANCE_THRESHOLD:
                        dominant_counts[max_term] += 1
                    
                    # Store this lag's result
                    lag_results.append({
                        'lag': lag,
                        'R_value': r,
                        'U1_value': u1,
                        'U2_value': u2,
                        'S_value': s,
                        'MI_value': mi,
                        'R_norm': r_norm,
                        'U1_norm': u1_norm,
                        'U2_norm': u2_norm,
                        'S_norm': s_norm
                    })
            except IndexError:
                print(f"Warning: Index out of bounds for lag {lag} (index {lag_idx}) for pair ({col1}, {col2}). Skipping lag.")
                continue
            except KeyError as e:
                print(f"Warning: Missing key {e} in pid_results for pair ({col1}, {col2}). Skipping dominance check.")
                break  # Stop checking lags for this pair if keys are missing
        
        # Check if we have enough valid lags to evaluate
        if total_valid_lags > 0:
            # Find term that is dominant across at least 90% of the lags
            for term, count in dominant_counts.items():
                dominance_ratio = count / total_valid_lags
                if dominance_ratio >= DOMINANCE_PERCENTAGE:
                    print(f"Found dominant term {term} for pair ({col1}, {col2}) across {dominance_ratio:.1%} of lags")
                    
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
                    
                    # Store this pair's result
                    dominant_pid_results.append({
                        'feature_pair': (col1, col2),
                        'dominant_term': term,
                        'dominance_ratio': dominance_ratio,
                        'lags_analyzed': total_valid_lags,
                        'avg_metrics': avg_metrics,
                        'lag_results': lag_results
                    })
                    break  # We've found the dominant term, no need to check others

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
        output_filename = f'pamap_subject{SUBJECT_ID}_lag{MAX_LAG}_bins{BINS}_thresh{DOMINANCE_THRESHOLD:.1f}_pct{int(DOMINANCE_PERCENTAGE*100)}.npy'
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
        print(f"Total feature pairs with dominant terms: {len(dominant_pid_results)}")
        for term, count in dominance_counts.items():
            print(f"  {term} dominant: {count} pairs")
        print("-------------------------")

    else:
        print("No dominant PID terms found with the current threshold and percentage criteria.")

if __name__ == "__main__":
    main()

