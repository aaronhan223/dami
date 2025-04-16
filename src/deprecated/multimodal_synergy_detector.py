import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pandas as pd
from sklearn.metrics import mutual_info_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import traceback

# Make sure we can import from src directory
sys.path.append('src')
try:
    from temporal_pid import temporal_pid, generate_causal_time_series
except ImportError:
    print("Error importing temporal_pid. Make sure the src directory exists and contains temporal_pid.py.")
    sys.exit(1)

def generate_multimodal_time_series(n_samples=300, synergy_segments=None, seed=None):
    """
    Generate synthetic multimodal time series with controllable synergy.
    
    Parameters:
    -----------
    n_samples : int, default=300
        Number of time points to generate
    synergy_segments : list, optional
        List of tuples (start, end) indicating high-synergy segments.
        If None, random segments will be created.
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X1, X2, Y : numpy.ndarray
        Generated time series
    true_segments : list
        List of segments with high synergy
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create empty time series
    X1 = np.zeros(n_samples)
    X2 = np.zeros(n_samples)
    Y = np.zeros(n_samples)
    
    # Generate base signals with noise
    for t in range(n_samples):
        X1[t] = 0.8 * np.sin(2 * np.pi * t / 50) + 0.2 * np.random.randn()
        X2[t] = 0.8 * np.cos(2 * np.pi * t / 50) + 0.2 * np.random.randn()
    
    # Define high synergy segments if not provided
    if synergy_segments is None:
        n_segments = np.random.randint(2, 4)
        synergy_segments = []
        
        for _ in range(n_segments):
            start = np.random.randint(0, n_samples - 50)
            length = np.random.randint(30, 50)
            end = min(start + length, n_samples)
            synergy_segments.append((start, end))
    
    # Default relationship: Y depends on X1 and X2 separately
    for t in range(1, n_samples):
        Y[t] = 0.3 * X1[t-1] + 0.3 * X2[t-1] + 0.1 * np.random.randn()
    
    # In high synergy segments: Y depends on X1 * X2 (interaction)
    for start, end in synergy_segments:
        for t in range(max(start+1, 1), end):
            # Overwrite with synergistic relationship
            Y[t] = 0.1 * X1[t-1] + 0.1 * X2[t-1] + 0.6 * X1[t-1] * X2[t-1] + 0.1 * np.random.randn()
    
    return X1, X2, Y, synergy_segments

def detect_synergistic_segments(X1, X2, Y, window_size=30, stride=15, lag=1, bins=3, min_synergy_threshold=0.1):
    """
    Detect time segments where multimodal synergy is highest.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Input modalities (time series)
    Y : numpy.ndarray
        Target variable (time series)
    window_size : int, default=30
        Size of sliding window for analysis
    stride : int, default=15
        Step size for sliding window
    lag : int, default=1
        Time lag for causal analysis
    bins : int, default=3
        Number of bins for discretization
    min_synergy_threshold : float, default=0.1
        Minimum synergy value to consider significant
            
    Returns:
    --------
    window_results : pandas.DataFrame
        DataFrame with PID results for each window
    high_synergy_segments : list
        List of tuples containing (start_idx, end_idx, synergy_value)
        for segments with high synergy
    """
    n_samples = len(Y)
    if len(X1) != n_samples or len(X2) != n_samples:
        raise ValueError("All time series must have the same length")
        
    print(f"Analyzing {n_samples} samples with window size {window_size} and stride {stride}")
    print(f"Using {bins} bins for discretization and lag={lag}")
        
    # Initialize storage for sliding window results
    window_results = {
        'window_start': [],
        'window_end': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': [],
        'synergy_ratio': []  # Synergy divided by total information
    }
    
    # Sliding window analysis
    total_windows = (n_samples - window_size) // stride + 1
    for i in range(0, n_samples - window_size, stride):
        # Print progress
        if i % (2 * stride) == 0:
            print(f"Processing window {i//stride + 1}/{total_windows}...")
            
        start_idx = i
        end_idx = i + window_size
        
        # Extract window data
        X1_window = X1[start_idx:end_idx].copy()
        X2_window = X2[start_idx:end_idx].copy()
        Y_window = Y[start_idx:end_idx].copy()
        
        try:
            # Calculate temporal PID for this window
            pid = temporal_pid(X1_window, X2_window, Y_window, lag=lag, bins=bins)
            
            # Store results
            window_results['window_start'].append(start_idx)
            window_results['window_end'].append(end_idx)
            window_results['redundancy'].append(pid['redundancy'])
            window_results['unique_x1'].append(pid['unique_x1'])
            window_results['unique_x2'].append(pid['unique_x2'])
            window_results['synergy'].append(pid['synergy'])
            window_results['total_di'].append(pid['total_di'])
            
            # Calculate synergy ratio (synergy relative to total information)
            if pid['total_di'] > 0:
                synergy_ratio = pid['synergy'] / pid['total_di']
            else:
                synergy_ratio = 0
            window_results['synergy_ratio'].append(synergy_ratio)
            
        except Exception as e:
            print(f"Error processing window {start_idx}-{end_idx}: {e}")
            # Add zeros for this window to maintain alignment
            window_results['window_start'].append(start_idx)
            window_results['window_end'].append(end_idx)
            window_results['redundancy'].append(0)
            window_results['unique_x1'].append(0)
            window_results['unique_x2'].append(0)
            window_results['synergy'].append(0)
            window_results['total_di'].append(0)
            window_results['synergy_ratio'].append(0)
    
    # Convert to pandas DataFrame
    results_df = pd.DataFrame(window_results)
    
    # Identify segments with high synergy
    high_synergy_mask = (results_df['synergy'] > min_synergy_threshold)
    high_synergy_segments = results_df[high_synergy_mask]
    
    # Convert to list of tuples
    high_synergy_list = [
        (row['window_start'], row['window_end'], row['synergy']) 
        for _, row in high_synergy_segments.iterrows()
    ]
    
    return results_df, high_synergy_list

def train_multimodal_model(X1, X2, Y, segments=None, task='classification', test_size=0.3, random_state=42):
    """
    Train models on different modality combinations to validate synergy insights.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Input modalities
    Y : numpy.ndarray
        Target variable
    segments : list or None
        List of time segments (start, end) to use. If None, use all data.
    task : str
        'classification' or 'regression'
    test_size : float
        Fraction of data to use for testing
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with model performance metrics
    """
    # Extract data from specified segments or use all data
    if segments:
        indices = []
        for start, end, _ in segments:
            indices.extend(range(start, end))
        indices = sorted(list(set(indices)))
        
        X1_data = X1[indices]
        X2_data = X2[indices]
        Y_data = Y[indices]
    else:
        X1_data = X1
        X2_data = X2
        Y_data = Y
    
    # Create modality combinations
    X1_only = X1_data.reshape(-1, 1)
    X2_only = X2_data.reshape(-1, 1)
    X_combined = np.column_stack([X1_data, X2_data])
    
    # For classification, convert Y to binary if continuous
    if task == 'classification' and np.issubdtype(Y_data.dtype, np.floating):
        Y_data = (Y_data > np.median(Y_data)).astype(int)
    
    # Results storage
    results = {}
    
    # Train and evaluate models for each modality combination
    for name, X in [('X1', X1_only), ('X2', X2_only), ('X1X2', X_combined)]:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y_data, test_size=test_size, random_state=random_state
        )
        
        # Train model
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1
            }
            
        else:  # regression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            
            model = RandomForestRegressor(n_estimators=50, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'r2_score': r2
            }
    
    return results

def visualize_results(results_df, high_synergy_segments, true_segments=None, model_results=None, save_path=None):
    """
    Visualize PID components, detected synergy segments, and model validation.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with PID results for each window
    high_synergy_segments : list
        List of tuples (start, end, synergy) for detected high synergy segments
    true_segments : list, optional
        List of tuples (start, end) for true high synergy segments
    model_results : dict, optional
        Dictionary with model performance metrics
    save_path : str, optional
        Path to save the figures. If None, figures are displayed but not saved.
    """
    # Create figures directory if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Time points for x-axis (midpoint of each window)
    time_points = np.array(results_df['window_start']) + (np.array(results_df['window_end']) - np.array(results_df['window_start'])) / 2
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Determine number of subplots based on available data
    n_plots = 2
    if model_results:
        n_plots = 3
    
    # Plot 1: PID components
    plt.subplot(n_plots, 1, 1)
    plt.plot(time_points, results_df['redundancy'], 'b-', label='Redundancy')
    plt.plot(time_points, results_df['unique_x1'], 'g-', label='Unique X1')
    plt.plot(time_points, results_df['unique_x2'], 'r-', label='Unique X2')
    plt.plot(time_points, results_df['synergy'], 'm-', linewidth=2, label='Synergy')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Information (bits)', fontsize=12)
    plt.title('Temporal PID Components Over Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Synergy with highlighted segments
    plt.subplot(n_plots, 1, 2)
    plt.plot(time_points, results_df['synergy'], 'm-', linewidth=2, label='Synergy')
    
    # Highlight detected high synergy regions
    for start, end, _ in high_synergy_segments:
        plt.axvspan(start, end, color='yellow', alpha=0.3)
    
    # Highlight true high synergy regions if provided
    if true_segments:
        for start, end in true_segments:
            plt.axvspan(start, end, color='green', alpha=0.2)
            
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Synergy (bits)', fontsize=12)
    plt.title('Detected High Synergy Segments', fontsize=14)
    if true_segments:
        plt.legend(['Synergy', 'Detected Segments', 'True Segments'])
    else:
        plt.legend(['Synergy', 'Detected Segments'])
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Model validation results if available
    if model_results:
        plt.subplot(n_plots, 1, 3)
        
        # Determine which metric to show based on task
        if 'accuracy' in model_results['X1']:
            # Classification metrics
            modalities = list(model_results.keys())
            x_pos = np.arange(len(modalities))
            accuracies = [model_results[mod]['accuracy'] for mod in modalities]
            f1_scores = [model_results[mod]['f1_score'] for mod in modalities]
            
            # Bar plot
            width = 0.35
            plt.bar(x_pos - width/2, accuracies, width, label='Accuracy')
            plt.bar(x_pos + width/2, f1_scores, width, label='F1 Score')
            
            plt.xlabel('Modality', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title('Model Performance Comparison (Classification)', fontsize=14)
            plt.xticks(x_pos, modalities)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(accuracies):
                plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
            for i, v in enumerate(f1_scores):
                plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
            
        else:
            # Regression metrics
            modalities = list(model_results.keys())
            x_pos = np.arange(len(modalities))
            r2_scores = [model_results[mod]['r2_score'] for mod in modalities]
            
            # Bar plot
            plt.bar(x_pos, r2_scores, label='R² Score')
            
            plt.xlabel('Modality', fontsize=12)
            plt.ylabel('R² Score', fontsize=12)
            plt.title('Model Performance Comparison (Regression)', fontsize=14)
            plt.xticks(x_pos, modalities)
            plt.grid(True, alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(r2_scores):
                plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_overview.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}_overview.png")
    else:
        plt.show()
    
    # Create heatmap of PID components if we have enough windows
    if len(results_df) > 3:
        plt.figure(figsize=(12, 6))
        pid_components = results_df[['redundancy', 'unique_x1', 'unique_x2', 'synergy']].values.T
        
        sns.heatmap(pid_components, cmap='viridis', 
                   xticklabels=False, yticklabels=['Redundancy', 'Unique X1', 'Unique X2', 'Synergy'],
                   cbar_kws={'label': 'Information (bits)'})
        
        plt.xlabel('Time Window', fontsize=12)
        plt.title('PID Components Across Time Windows', fontsize=14)
        
        if save_path:
            plt.savefig(f"{save_path}_heatmap.png", dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}_heatmap.png")
        else:
            plt.show()

def main():
    """Main function to demonstrate multimodal synergy detection."""
    try:
        print("\n=== Multimodal Synergy Detection ===\n")
        
        # Set random seed
        np.random.seed(42)
        
        # 1. Generate synthetic data with known synergy segments
        print("Generating synthetic multimodal time series...")
        true_synergy_segments = [
            (50, 80),
            (150, 180),
            (250, 280)
        ]
        X1, X2, Y, _ = generate_multimodal_time_series(
            n_samples=300, 
            synergy_segments=true_synergy_segments,
            seed=42
        )
        
        print(f"Generated {len(X1)} time points with {len(true_synergy_segments)} high-synergy segments")
        
        # 2. Detect segments with high synergy
        print("\nDetecting time segments with high multimodal synergy...")
        results_df, detected_segments = detect_synergistic_segments(
            X1, X2, Y,
            window_size=30,
            stride=15,
            lag=1,
            bins=3,
            min_synergy_threshold=0.1
        )
        
        print(f"\nDetected {len(detected_segments)} segments with high synergy:")
        for i, (start, end, synergy) in enumerate(detected_segments[:5]):
            print(f"  Segment {i+1}: Time {start}-{end}, Synergy: {synergy:.4f}")
        
        # 3. Train models to validate insights
        print("\nValidating insights with models...")
        # Convert Y to binary for classification
        Y_binary = (Y > np.median(Y)).astype(int)
        
        # Train on high synergy segments
        high_synergy_results = train_multimodal_model(
            X1, X2, Y_binary, 
            segments=detected_segments, 
            task='classification'
        )
        
        print("\nModel performance on high synergy segments:")
        for modality, metrics in high_synergy_results.items():
            print(f"  {modality}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        
        # 4. Visualize results
        print("\nVisualizing results...")
        visualize_results(
            results_df, 
            detected_segments, 
            true_segments=true_synergy_segments,
            model_results=high_synergy_results,
            save_path='results/multimodal_synergy'
        )
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 