import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pdb # Keep pdb for potential debugging

# Try to import cvxpy for handling potential solver exceptions
try:
    import cvxpy as cp
except ImportError:
    # It's okay if this fails as temporal_pid.py might handle cvxpy internally
    print("Note: cvxpy not found, but may not be needed directly in this script.")
    cp = None

# Assuming temporal_pid.py is in the same directory or accessible in PYTHONPATH
try:
    from temporal_pid import (
        create_probability_distribution,
        solve_Q_temporal,
        CoI_temporal,
        UI_temporal,
        CI_temporal,
        MI,
        # We won't use this anymore
        # generate_causal_time_series 
    )
except ImportError:
    print("Error: temporal_pid.py not found. Please place it in the same directory or add it to your PYTHONPATH.")
    exit()

# --- Configuration ---
WINDOW_SIZE = 100  # Size of the sliding window
STEP_SIZE = 10     # Step for the sliding window
PID_LAG = 1        # Lag for PID calculation
PID_BINS = 5       # Bins for PID discretization (keep small for performance/accuracy)
SYNERGY_THRESHOLD_PERCENTILE = 90 # Identify segments in the top % of synergy
MODEL_LAG = 1      # Lag for predictive models
N_SAMPLES = 2000   # Number of samples for synthetic data
RESULTS_DIR = './results_synergy' # Directory to save results

# --- Helper Functions ---

def generate_synergistic_time_series(n_samples=2000, seed=442):
    """
    Generate synthetic time series data with explicit synergistic regions that are 
    designed to be more easily detected by temporal PID analysis.
    
    This function creates:
    1. X1: A signal with sinusoidal patterns and controlled noise
    2. X2: A signal with different frequency patterns and controlled noise
    3. Y: A target variable with clearly differentiated regions:
       - Linear X1 dependency regions
       - Linear X2 dependency regions
       - Strong nonlinear synergistic regions (multiplicative and XOR-like)
       - Transition regions with reduced noise
       
    Returns:
        X1, X2, Y (numpy arrays), synergy_periods (list of tuples with start/end indices)
    """
    np.random.seed(seed)
    
    # Create time vector with more samples for better resolution
    t = np.linspace(0, 10, n_samples)
    
    # Generate base signals with cleaner patterns (reduced noise)
    X1 = np.sin(0.5 * t) + 0.1 * np.random.randn(n_samples)  # Slower oscillation with less noise
    X2 = np.sin(2.0 * t) + np.cos(0.3 * t) + 0.1 * np.random.randn(n_samples)  # More complex pattern
    
    # Add some autocorrelation to make signals more realistic
    for i in range(3, n_samples):
        X1[i] = 0.3 * X1[i] + 0.7 * X1[i-1] + 0.05 * np.random.randn()
        X2[i] = 0.3 * X2[i] + 0.7 * X2[i-1] + 0.05 * np.random.randn()
    
    # Initialize target variable
    Y = np.zeros(n_samples)
    
    # Track actual synergistic periods for validation
    synergy_periods = []
    
    # Create distinct regions with different dependency patterns and minimal overlap
    
    # Region 1: Y depends primarily on X1 (200-450)
    start, end = 200, 450
    Y[start:end] = 1.0 * X1[start:end] + 0.05 * np.random.randn(end-start)
    
    # Buffer region (450-500) - mild dependency on both to create a transition
    start, end = 450, 500
    Y[start:end] = 0.5 * X1[start:end] + 0.2 * X2[start:end] + 0.1 * np.random.randn(end-start)
    
    # Region 2: Y depends primarily on X2 (500-750)
    start, end = 500, 750
    Y[start:end] = 1.0 * X2[start:end] + 0.05 * np.random.randn(end-start)
    
    # Buffer region (750-800) - mild dependency on both
    start, end = 750, 800
    Y[start:end] = 0.2 * X1[start:end] + 0.5 * X2[start:end] + 0.1 * np.random.randn(end-start)
    
    # Region 3: SYNERGISTIC - Multiplicative relationship (800-1000)
    # Stronger effect, very low noise to make pattern clear
    start, end = 800, 1000
    Y[start:end] = 1.2 * X1[start:end] * X2[start:end] + 0.03 * np.random.randn(end-start)
    synergy_periods.append((start, end))
    
    # Buffer region (1000-1050)
    start, end = 1000, 1050
    Y[start:end] = 0.3 * X1[start:end] + 0.3 * X2[start:end] + 0.1 * np.random.randn(end-start)
    
    # Region 4: Y depends on X1 again but with negative relationship (1050-1250)
    start, end = 1050, 1250
    Y[start:end] = -0.9 * X1[start:end] + 0.05 * np.random.randn(end-start)
    
    # Buffer region (1250-1300)
    start, end = 1250, 1300
    Y[start:end] = -0.4 * X1[start:end] + 0.3 * X2[start:end] + 0.1 * np.random.randn(end-start)
    
    # Region 5: SYNERGISTIC - XOR-like (absolute difference) relationship (1300-1500)
    # Increased coefficient and reduced noise for clearer pattern
    start, end = 1300, 1500
    Y[start:end] = 1.5 * np.abs(X1[start:end] - X2[start:end]) + 0.03 * np.random.randn(end-start)
    synergy_periods.append((start, end))
    
    # Region 6: New SYNERGISTIC pattern - quadratic interaction (1600-1800)
    start, end = 1600, 1800
    Y[start:end] = 1.0 * (X1[start:end]**2 + X2[start:end]**2) + 0.5 * X1[start:end] * X2[start:end] + 0.03 * np.random.randn(end-start)
    synergy_periods.append((start, end))
    
    # Create transition mask for all specified regions
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in [(200, 450), (450, 500), (500, 750), (750, 800), 
                       (800, 1000), (1000, 1050), (1050, 1250), (1250, 1300), 
                       (1300, 1500), (1600, 1800)]:
        mask[start:end] = True
    
    # Create smoother transitions for non-specified regions
    for i in range(n_samples):
        if not mask[i]:
            # Find nearest specified point
            nearest_idxs = np.where(mask)[0]
            if len(nearest_idxs) > 0:
                nearest_specified = np.argmin(np.abs(nearest_idxs - i))
                nearest_idx = nearest_idxs[nearest_specified]
                distance = abs(nearest_idx - i)
                
                # Use smoother transition with exponential decay
                decay = np.exp(-distance / 30)  # Faster decay (30 instead of 50)
                # Use a balanced mix for transition regions with lower noise
                Y[i] = decay * Y[nearest_idx] + (1-decay) * (0.3 * X1[i] + 0.3 * X2[i] + 0.2 * np.random.randn())
    
    # Apply some mild smoothing to avoid sharp transitions between regions
    # This helps the PID analysis which uses sliding windows
    Y_smoothed = np.copy(Y)
    window_size = 5
    for i in range(window_size, n_samples-window_size):
        Y_smoothed[i] = 0.7 * Y[i] + 0.3 * np.mean(Y[i-window_size:i+window_size])
    Y = Y_smoothed
    
    # Normalize to similar ranges for better visualization and analysis
    # Use robust scaling to avoid influence of outliers
    X1 = (X1 - np.median(X1)) / (np.percentile(X1, 75) - np.percentile(X1, 25))
    X2 = (X2 - np.median(X2)) / (np.percentile(X2, 75) - np.percentile(X2, 25))
    Y = (Y - np.median(Y)) / (np.percentile(Y, 75) - np.percentile(Y, 25))
    
    return X1, X2, Y, synergy_periods

def calculate_pid_for_segment(x1_segment, x2_segment, y_segment, lag, bins):
    """Calculates PID components for a given data segment."""
    # Check if segments are long enough for the specified lag
    min_len = lag + 1
    if len(x1_segment) < min_len or len(x2_segment) < min_len or len(y_segment) < min_len:
        # print(f"Warning: Segment length ({len(x1_segment)}) too short for lag ({lag}). Skipping.")
        return {'redundancy': np.nan, 'unique_x1': np.nan, 'unique_x2': np.nan, 'synergy': np.nan, 'total_di': np.nan}

    try:
        P = create_probability_distribution(x1_segment, x2_segment, y_segment, lag, bins)

        # Check for near-zero probabilities which can cause issues in KL divergence
        if np.any(P == 0):
            # print("Adding epsilon smoothing to P")
            P += 1e-12 # Add small epsilon
            P /= np.sum(P) # Renormalize

        # Check if P is valid (sums to 1)
        if not np.isclose(np.sum(P), 1.0):
            print(f"Warning: Probability distribution P does not sum to 1 (sum={np.sum(P)}). Skipping segment.")
            return {'redundancy': np.nan, 'unique_x1': np.nan, 'unique_x2': np.nan, 'synergy': np.nan, 'total_di': np.nan}

        Q = solve_Q_temporal(P)

        # Check if optimization was successful
        if Q is None or np.isnan(Q).any():
            print(f"Warning: Optimization for Q failed or returned NaN. Skipping segment.")
            return {'redundancy': np.nan, 'unique_x1': np.nan, 'unique_x2': np.nan, 'synergy': np.nan, 'total_di': np.nan}
             
        # Add epsilon smoothing to Q as well before calculating CoI, UI
        if np.any(Q == 0):
            # print("Adding epsilon smoothing to Q")
            Q += 1e-12 # Add small epsilon
            Q /= np.sum(Q) # Renormalize
            # Verify Q still meets constraints (approximately) after smoothing - this is tricky.
            # For now, we proceed, but this could introduce slight inaccuracies.

        redundancy = CoI_temporal(Q)
        unique_x1 = UI_temporal(Q, cond_id=1)
        unique_x2 = UI_temporal(Q, cond_id=0)
        synergy = CI_temporal(P, Q) # Uses P and Q

        # Ensure non-negative results (numerical issues can cause small negatives)
        redundancy = max(0, redundancy)
        unique_x1 = max(0, unique_x1)
        unique_x2 = max(0, unique_x2)
        synergy = max(0, synergy)

        # Recalculate Total DI directly from P for comparison
        try:
            total_di = MI(P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1])))
        except Exception as mi_e:
            print(f"Warning: Could not calculate total MI from P: {mi_e}")
            total_di = np.nan

        # Check consistency
        calc_sum = redundancy + unique_x1 + unique_x2 + synergy
        if not np.isnan(total_di) and not np.isclose(total_di, calc_sum, atol=1e-3): # Allow some tolerance
            print(f"Warning: PID components sum ({calc_sum:.4f}) != Total DI ({total_di:.4f}). Diff: {abs(total_di-calc_sum):.4f}")
            # This might happen due to numerical precision or issues in calculation.

        return {
            'redundancy': redundancy,
            'unique_x1': unique_x1,
            'unique_x2': unique_x2,
            'synergy': synergy,
            'total_di': total_di # Report the directly calculated total DI
        }
    except Exception as e:
        if cp and hasattr(cp, 'error') and hasattr(cp.error, 'SolverError') and isinstance(e, cp.error.SolverError):
            print(f"CVXPY SolverError during PID calculation: {e}. Skipping segment.")
        elif isinstance(e, ValueError):
            print(f"ValueError during PID calculation: {e}. Skipping segment.")
        else:
            print(f"Unexpected error during PID calculation: {e}. Skipping segment.")
            # import traceback # Uncomment for detailed trace
            # traceback.print_exc()
        return {'redundancy': np.nan, 'unique_x1': np.nan, 'unique_x2': np.nan, 'synergy': np.nan, 'total_di': np.nan}


def create_lagged_features(data, lag):
    """Creates lagged features for modeling."""
    df = pd.DataFrame(data)
    # Shift introduces NaNs at the beginning
    lagged_df = df.shift(lag)
    # Return numpy array, NaNs will be handled by train/test split or imputation if needed
    return lagged_df.values 

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load Data (Using synthetic data for now)
    print("Generating synthetic time series data with explicit synergistic regions...")
    # TODO: Replace this with your actual data loading mechanism
    # Example:
    # data = pd.read_csv('your_data.csv')
    # X1 = data['series1'].values
    # X2 = data['series2'].values
    # Y = data['target'].values
    # N_SAMPLES = len(Y)
    # times = data['time_index'].values or np.arange(N_SAMPLES)
    X1, X2, Y, synergy_periods = generate_synergistic_time_series(n_samples=N_SAMPLES, seed=42)
    times = np.arange(N_SAMPLES)
    
    # Create a mask for the true synergistic periods (for validation)
    true_synergy_mask = np.zeros(N_SAMPLES, dtype=bool)
    for start, end in synergy_periods:
        true_synergy_mask[start:end] = True
    
    print(f"Generated data with {len(synergy_periods)} known synergistic periods:")
    for i, (start, end) in enumerate(synergy_periods):
        print(f"  Period {i+1}: Indices {start}-{end} (Duration: {end-start} samples)")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2. Calculate Temporal PID using Sliding Window
    print(f"Calculating PID with sliding window (size={WINDOW_SIZE}, step={STEP_SIZE})...")
    pid_results = []
    window_centers = []

    for i in range(0, N_SAMPLES - WINDOW_SIZE + 1, STEP_SIZE):
        window_end = i + WINDOW_SIZE
        window_center_time = times[i + WINDOW_SIZE // 2] # Use time index if available

        # Extract segments
        x1_segment = X1[i:window_end]
        x2_segment = X2[i:window_end]
        y_segment = Y[i:window_end]

        # Calculate PID for the current window
        pid_values = calculate_pid_for_segment(x1_segment, x2_segment, y_segment, PID_LAG, PID_BINS)

        # Store results if valid
        pid_results.append(pid_values) # Append even if NaN to keep index alignment initially
        window_centers.append(window_center_time)


    if not window_centers: # Check if any windows were processed
        print("Error: No windows processed. Check N_SAMPLES and WINDOW_SIZE.")
        exit()

    # Create DataFrame and handle potential NaNs from calculation errors
    pid_df = pd.DataFrame(pid_results, index=window_centers)
    pid_df.index.name = 'time_center'

    print(f"Initial PID calculation yielded {pid_df.shape[0]} windows.")
    print(f"Number of windows with NaN synergy: {pid_df['synergy'].isna().sum()}")

    # Interpolate PID values to cover the entire time range for visualization and analysis
    # Create a full time index based on original data
    full_time_index = times
    pid_df_interpolated = pid_df.reindex(full_time_index)
    # First, interpolate linearly for smoother transitions
    pid_df_interpolated = pid_df_interpolated.interpolate(method='linear')
    # Then, fill remaining NaNs (at ends) with the nearest value
    pid_df_interpolated = pid_df_interpolated.fillna(method='ffill').fillna(method='bfill')

    # Ensure no NaNs remain in key columns after interpolation/filling
    if pid_df_interpolated[['redundancy', 'unique_x1', 'unique_x2', 'synergy']].isna().any().any():
        print("Warning: NaNs remain in PID dataframe after interpolation. Filling with 0.")
        pid_df_interpolated.fillna(0, inplace=True)


    # 3. Identify Synergistic Segments based on interpolated data
    print("Identifying synergistic segments...")
    # Calculate threshold on the original, non-interpolated, non-NaN synergy values
    valid_synergy = pid_df['synergy'].dropna()
    if valid_synergy.empty:
        print("Warning: No valid synergy values calculated. Cannot determine threshold.")
        synergy_threshold = 0
        pid_df_interpolated['is_synergistic'] = False
    else:
        synergy_threshold = np.percentile(valid_synergy, SYNERGY_THRESHOLD_PERCENTILE)
        pid_df_interpolated['is_synergistic'] = pid_df_interpolated['synergy'] >= synergy_threshold
        
    print(f"Synergy threshold (>{SYNERGY_THRESHOLD_PERCENTILE} percentile of valid windows): {synergy_threshold:.4f}")
    print(f"Identified {pid_df_interpolated['is_synergistic'].sum()} synergistic time points in interpolated data.")


    # 4. Train Predictive Models & Validate
    print(f"Training predictive models (lag={MODEL_LAG})...")

    # Prepare lagged data for modeling
    X1_lagged = create_lagged_features(X1, MODEL_LAG)
    X2_lagged = create_lagged_features(X2, MODEL_LAG)
    X_combined_lagged = np.column_stack((X1_lagged, X2_lagged))
    target_Y = Y # Target variable Y

    # Define train/test split indices, accounting for the lag
    # We cannot use the first `MODEL_LAG` samples for training or testing features
    valid_indices = np.arange(MODEL_LAG, N_SAMPLES)
    
    if len(valid_indices) < 20: # Need sufficient data points after lagging
        print("Error: Not enough data points after applying model lag for train/test split.")
        exit()
        
    # Use train_test_split on the valid indices
    train_indices, test_indices = train_test_split(
        valid_indices, test_size=0.3, shuffle=False # Keep temporal order
    )

    print(f"Training on indices {train_indices.min()} to {train_indices.max()} ({len(train_indices)} points)")
    print(f"Testing on indices {test_indices.min()} to {test_indices.max()} ({len(test_indices)} points)")

    # Check if train/test sets are too small
    if len(train_indices) < 10 or len(test_indices) < 10:
       print("Warning: Train or test set size is very small. Results may be unreliable.")

    # --- Train Models ---
    # Model 1: Y ~ X1
    print("Training Model 1 (Y ~ X1)...")
    model1 = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model1.fit(X1_lagged[train_indices], target_Y[train_indices])
    preds1 = model1.predict(X1_lagged[test_indices])
    errors1 = (target_Y[test_indices] - preds1) ** 2

    # Model 2: Y ~ X2
    print("Training Model 2 (Y ~ X2)...")
    model2 = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model2.fit(X2_lagged[train_indices], target_Y[train_indices])
    preds2 = model2.predict(X2_lagged[test_indices])
    errors2 = (target_Y[test_indices] - preds2) ** 2

    # Model 3: Y ~ X1 + X2
    print("Training Model 3 (Y ~ X1 + X2)...")
    model3 = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model3.fit(X_combined_lagged[train_indices], target_Y[train_indices])
    preds3 = model3.predict(X_combined_lagged[test_indices])
    errors3 = (target_Y[test_indices] - preds3) ** 2

    # Align errors with the original time index and PID results for the test period
    # Use the test_indices which correspond to the original time axis
    error_df = pd.DataFrame({
        'error1': errors1,
        'error2': errors2,
        'error3': errors3
    }, index=test_indices) # Index should align with pid_df_interpolated

    # Merge PID info with errors for the test period
    # Use pid_df_interpolated which has the full time index
    analysis_df = pid_df_interpolated.loc[test_indices].join(error_df)
    analysis_df.dropna(inplace=True) # Drop rows where errors couldn't be calculated if any

    if analysis_df.empty:
        print("Error: No overlapping data between PID results and model evaluation after join/dropna.")
        # This might happen if test_indices has no match in pid_df_interpolated's index
        print(f"Test indices range: {test_indices.min()} - {test_indices.max()}")
        print(f"PID interpolated index range: {pid_df_interpolated.index.min()} - {pid_df_interpolated.index.max()}")
        exit()

    # 5. Validation: Compare errors in synergistic vs. non-synergistic segments
    print("Validating model performance in synergistic segments...")

    synergistic_errors = analysis_df[analysis_df['is_synergistic']]
    non_synergistic_errors = analysis_df[~analysis_df['is_synergistic']]

    print(f"Number of test points in synergistic segments: {len(synergistic_errors)}")
    print(f"Number of test points in non-synergistic segments: {len(non_synergistic_errors)}")

    if synergistic_errors.empty or non_synergistic_errors.empty:
         print("Warning: Could not compare segments - one type is empty in the test set.")
    else:
        mse_syn = synergistic_errors[['error1', 'error2', 'error3']].mean()
        mse_non_syn = non_synergistic_errors[['error1', 'error2', 'error3']].mean()

        print("\n--- Model Performance Comparison (Mean Squared Error on Test Set) ---")
        print("Segment         | MSE Model 1 (X1) | MSE Model 2 (X2) | MSE Model 3 (X1+X2)")
        print("----------------|------------------|------------------|--------------------")
        # Check for NaN before formatting
        err1_syn_str = f"{mse_syn.get('error1', np.nan):.4f}"
        err2_syn_str = f"{mse_syn.get('error2', np.nan):.4f}"
        err3_syn_str = f"{mse_syn.get('error3', np.nan):.4f}"
        err1_non_str = f"{mse_non_syn.get('error1', np.nan):.4f}"
        err2_non_str = f"{mse_non_syn.get('error2', np.nan):.4f}"
        err3_non_str = f"{mse_non_syn.get('error3', np.nan):.4f}"

        print(f"Synergistic     | {err1_syn_str:<16} | {err2_syn_str:<16} | {err3_syn_str}")
        print(f"Non-Synergistic | {err1_non_str:<16} | {err2_non_str:<16} | {err3_non_str}")


        # Calculate relative improvement of Model 3 vs the BEST single model in each segment type
        best_single_syn_mse = min(mse_syn.get('error1', np.inf), mse_syn.get('error2', np.inf))
        best_single_non_syn_mse = min(mse_non_syn.get('error1', np.inf), mse_non_syn.get('error2', np.inf))

        improvement_syn = np.nan
        if best_single_syn_mse != np.inf and best_single_syn_mse > 0 and not np.isnan(mse_syn.get('error3')):
           improvement_syn = (best_single_syn_mse - mse_syn['error3']) / best_single_syn_mse

        improvement_non_syn = np.nan
        if best_single_non_syn_mse != np.inf and best_single_non_syn_mse > 0 and not np.isnan(mse_non_syn.get('error3')):
            improvement_non_syn = (best_single_non_syn_mse - mse_non_syn['error3']) / best_single_non_syn_mse

        print("\nRelative MSE Improvement of Combined Model (vs. Best Single):")
        if not np.isnan(improvement_syn):
            print(f"  In Synergistic Segments: {improvement_syn:.2%}")
        else:
            print("  In Synergistic Segments: N/A")
        if not np.isnan(improvement_non_syn):
             print(f"  In Non-Synergistic Segments: {improvement_non_syn:.2%}")
        else:
             print("  In Non-Synergistic Segments: N/A")

        if not np.isnan(improvement_syn) and not np.isnan(improvement_non_syn):
            if improvement_syn > improvement_non_syn:
                print("\nValidation check: Combined model shows greater relative improvement in synergistic segments.")
            elif improvement_syn < improvement_non_syn:
                 print("\nValidation check: Combined model shows LESS relative improvement in synergistic segments.")
            else:
                 print("\nValidation check: Combined model shows similar relative improvement in both segment types.")
        else:
            print("\nValidation check: Could not compare relative improvements due to NaN values.")


    # 6. Visualization
    print("Generating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True) # Increased height

    # Plot 1: PID Components (using interpolated data for full view)
    ax = axes[0]
    pid_plot_df = pid_df_interpolated # Use the interpolated df for plotting
    ax.plot(pid_plot_df.index, pid_plot_df['redundancy'], label='Redundancy (R)', color='blue', alpha=0.8)
    ax.plot(pid_plot_df.index, pid_plot_df['unique_x1'], label='Unique X1 (U1)', color='green', alpha=0.8)
    ax.plot(pid_plot_df.index, pid_plot_df['unique_x2'], label='Unique X2 (U2)', color='red', alpha=0.8)
    ax.plot(pid_plot_df.index, pid_plot_df['synergy'], label='Synergy (S)', color='magenta', linewidth=2)
    # ax.plot(pid_plot_df.index, pid_plot_df['total_di'], label='Total DI', color='black', linestyle='--', alpha=0.7) # Optional: Can make plot busy
    ax.set_ylabel('Information (bits)')
    ax.set_title(f'Temporal PID Components (Window={WINDOW_SIZE}, Lag={PID_LAG}, Bins={PID_BINS})')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Highlight synergistic segments on PID plot
    min_val_pid, max_val_pid = ax.get_ylim()
    # Ensure where condition is boolean and aligns with index
    where_synergistic = pid_plot_df['is_synergistic'].astype(bool) 
    ax.fill_between(pid_plot_df.index, min_val_pid, max_val_pid, where=where_synergistic,
                    facecolor='magenta', alpha=0.15, label=f'Detected Synergy (Top {100-SYNERGY_THRESHOLD_PERCENTILE}%)')
    
    # Highlight true synergistic periods with a different color/pattern
    ax.fill_between(times, min_val_pid, max_val_pid, where=true_synergy_mask,
                    facecolor='orange', alpha=0.15, hatch='\\', label='True Synergy')
    
    ax.set_ylim(min_val_pid, max_val_pid) # Re-apply ylim after fill_between
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    # Customize legend order if needed
    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=9)


    # Plot 2: Model Prediction Errors (only for the test set)
    ax = axes[1]
    # Use analysis_df which contains errors and synergy info for the test period
    error_plot_df = analysis_df
    rolling_window_size = max(1, min(50, len(error_plot_df) // 10)) # Adaptive rolling window, at least 1

    ax.plot(error_plot_df.index, error_plot_df['error1'].rolling(rolling_window_size, min_periods=1).mean(), label='Error M1 (X1)', color='green', alpha=0.8)
    ax.plot(error_plot_df.index, error_plot_df['error2'].rolling(rolling_window_size, min_periods=1).mean(), label='Error M2 (X2)', color='red', alpha=0.8)
    ax.plot(error_plot_df.index, error_plot_df['error3'].rolling(rolling_window_size, min_periods=1).mean(), label='Error M3 (X1+X2)', color='purple', linewidth=2)
    ax.set_ylabel('Mean Squared Error (Rolling)')
    ax.set_title(f'Predictive Model Performance (Lag={MODEL_LAG}, Test Set, Rolling Window={rolling_window_size})')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0) # Errors cannot be negative

    # Highlight synergistic segments on Error plot
    min_val_err, max_val_err = ax.get_ylim()
    # Ensure where condition aligns with error_plot_df index
    where_synergistic_test = error_plot_df['is_synergistic'].astype(bool)
    ax.fill_between(error_plot_df.index, min_val_err, max_val_err, where=where_synergistic_test,
                    facecolor='magenta', alpha=0.15, label=f'Detected Synergy (Top {100-SYNERGY_THRESHOLD_PERCENTILE}%)')
    
    # Highlight true synergistic periods that fall within the test period
    test_mask = np.zeros(N_SAMPLES, dtype=bool)
    test_mask[test_indices] = True
    test_true_synergy_mask = np.logical_and(true_synergy_mask, test_mask)
    ax.fill_between(times, min_val_err, max_val_err, where=test_true_synergy_mask,
                    facecolor='orange', alpha=0.15, hatch='\\', label='True Synergy')
    
    ax.set_ylim(min_val_err, max_val_err) # Re-apply ylim
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=9)


    # Plot 3: Raw Time Series Data
    ax = axes[2]
    ax.plot(times, X1, label='X1', color='green', alpha=0.6)
    ax.plot(times, X2, label='X2', color='red', alpha=0.6)
    # Plot Y on a secondary axis if scales differ significantly
    ax_y = ax.twinx()
    ax_y.plot(times, Y, label='Y (Target)', color='black', alpha=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('X1 / X2 Value', color='gray')
    ax_y.set_ylabel('Y Value', color='black')
    ax.set_title('Input and Target Time Series')
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_y.get_legend_handles_labels()
    ax_y.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Highlight synergistic segments on Data plot
    min_val_data, max_val_data = ax.get_ylim()
    # Detected synergy
    where_synergistic_full = pid_df_interpolated['is_synergistic'].astype(bool)
    ax.fill_between(pid_df_interpolated.index, min_val_data, max_val_data, where=where_synergistic_full,
                    facecolor='magenta', alpha=0.15, label='Detected Synergy')
    # True synergy
    ax.fill_between(times, min_val_data, max_val_data, where=true_synergy_mask,
                    facecolor='orange', alpha=0.15, hatch='\\', label='True Synergy')
    ax.set_ylim(min_val_data, max_val_data)

    # Add evaluation metric for detection accuracy
    # Calculate intersection over union (IoU) between true and detected synergy
    intersection = np.logical_and(where_synergistic_full, true_synergy_mask).sum()
    union = np.logical_or(where_synergistic_full, true_synergy_mask).sum()
    iou = intersection / union if union > 0 else 0
    
    # Calculate precision and recall
    precision = intersection / where_synergistic_full.sum() if where_synergistic_full.sum() > 0 else 0
    recall = intersection / true_synergy_mask.sum() if true_synergy_mask.sum() > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add detection accuracy metrics to the plot
    plt.figtext(0.5, 0.01, 
                f"Synergy Detection Metrics: IoU={iou:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}",
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
    plot_filename = os.path.join(RESULTS_DIR, 'synergy_analysis_results.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Visualization saved to {plot_filename}")
    # plt.show() # Uncomment to display plot interactively

    # 7. Additional Analysis: Compare model performance specifically within true synergy regions
    print("\nComparing model performance in TRUE synergy regions vs other regions:")
    
    # Create masks for the test indices within true synergy regions and outside them
    test_in_true_synergy = np.intersect1d(test_indices, np.where(true_synergy_mask)[0])
    test_outside_true_synergy = np.setdiff1d(test_indices, test_in_true_synergy)
    
    if len(test_in_true_synergy) > 0 and len(test_outside_true_synergy) > 0:
        # Get errors within true synergy regions
        errors_in_true_synergy = pd.DataFrame({
            'error1': (target_Y[test_in_true_synergy] - model1.predict(X1_lagged[test_in_true_synergy])) ** 2,
            'error2': (target_Y[test_in_true_synergy] - model2.predict(X2_lagged[test_in_true_synergy])) ** 2,
            'error3': (target_Y[test_in_true_synergy] - model3.predict(X_combined_lagged[test_in_true_synergy])) ** 2
        })
        
        # Get errors outside true synergy regions
        errors_outside_true_synergy = pd.DataFrame({
            'error1': (target_Y[test_outside_true_synergy] - model1.predict(X1_lagged[test_outside_true_synergy])) ** 2,
            'error2': (target_Y[test_outside_true_synergy] - model2.predict(X2_lagged[test_outside_true_synergy])) ** 2,
            'error3': (target_Y[test_outside_true_synergy] - model3.predict(X_combined_lagged[test_outside_true_synergy])) ** 2
        })
        
        # Calculate mean errors
        mse_true_syn = errors_in_true_synergy.mean()
        mse_outside_syn = errors_outside_true_synergy.mean()
        
        print("\n--- Model Performance in TRUE Synergy Regions (Mean Squared Error) ---")
        print("Region           | MSE Model 1 (X1) | MSE Model 2 (X2) | MSE Model 3 (X1+X2)")
        print("-----------------|------------------|------------------|--------------------")
        print(f"TRUE Synergy     | {mse_true_syn['error1']:.4f} | {mse_true_syn['error2']:.4f} | {mse_true_syn['error3']:.4f}")
        print(f"Non-Synergy      | {mse_outside_syn['error1']:.4f} | {mse_outside_syn['error2']:.4f} | {mse_outside_syn['error3']:.4f}")
        
        # Calculate relative improvement within true synergy regions
        best_single_true_syn = min(mse_true_syn['error1'], mse_true_syn['error2'])
        best_single_outside_syn = min(mse_outside_syn['error1'], mse_outside_syn['error2'])
        
        improvement_true_syn = (best_single_true_syn - mse_true_syn['error3']) / best_single_true_syn
        improvement_outside_syn = (best_single_outside_syn - mse_outside_syn['error3']) / best_single_outside_syn
        
        print("\nRelative Improvement Using Combined Model (vs Best Single):")
        print(f"  In TRUE Synergy Regions: {improvement_true_syn:.2%}")
        print(f"  In Non-Synergy Regions:  {improvement_outside_syn:.2%}")
        
        if improvement_true_syn > improvement_outside_syn:
            print("\nValidation check: Combined model shows greater improvement in TRUE synergy regions ✓")
            print(f"  Improvement differential: {improvement_true_syn - improvement_outside_syn:.2%}")
        else:
            print("\nNote: Combined model does NOT show greater improvement in TRUE synergy regions")
            print(f"  Improvement differential: {improvement_true_syn - improvement_outside_syn:.2%}")
    else:
        print("  Cannot perform analysis: No test samples in one or both region types")

    print("\nAnalysis complete.") 