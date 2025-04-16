import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def generate_multimodal_time_series(n_samples=300, synergy_segments=None, seed=None):
    """Generate synthetic multimodal time series with controllable synergy"""
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
        for t in range(max(start+1, 1), min(end, n_samples)):
            # Overwrite with synergistic relationship
            Y[t] = 0.1 * X1[t-1] + 0.1 * X2[t-1] + 0.6 * X1[t-1] * X2[t-1] + 0.1 * np.random.randn()
    
    return X1, X2, Y, synergy_segments

def simulate_pid_results(X1, X2, Y, window_size=30, stride=15, min_synergy_threshold=0.1):
    """
    Simulate PID results for demonstration purposes.
    In a real implementation, this would call temporal_pid for each window.
    """
    n_samples = len(Y)
    n_windows = (n_samples - window_size) // stride + 1
    
    # Storage for results
    window_results = {
        'window_start': [],
        'window_end': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': [],
        'synergy_ratio': []
    }
    
    # For demonstration, generate synthetic PID values that align with the
    # synergistic relationship in the data
    for i in range(0, n_samples - window_size, stride):
        window_start = i
        window_end = i + window_size
        window_mid = (window_start + window_end) // 2
        
        # Extract window
        X1_window = X1[window_start:window_end]
        X2_window = X2[window_start:window_end]
        Y_window = Y[window_start:window_end]
        
        # Calculate a simple proxy for synergy based on X1*X2 correlation with Y
        X1X2_product = X1_window * X2_window
        
        # Calculate correlations as proxies for information measures
        corr_X1Y = abs(np.corrcoef(X1_window, Y_window)[0, 1])
        corr_X2Y = abs(np.corrcoef(X2_window, Y_window)[0, 1])
        corr_X1X2Y = abs(np.corrcoef(X1X2_product, Y_window)[0, 1])
        
        # Simulate PID components
        # Higher synergy when X1*X2 correlates more strongly with Y
        redundancy = min(corr_X1Y, corr_X2Y) * 0.5
        unique_x1 = max(0, corr_X1Y - redundancy) * 0.7
        unique_x2 = max(0, corr_X2Y - redundancy) * 0.7
        
        # Synergy is high when product correlates better than individual variables
        synergy = max(0, corr_X1X2Y - corr_X1Y - corr_X2Y) * 2.0
        
        # Add some random noise to make it look more realistic
        redundancy += 0.05 * np.random.randn()
        unique_x1 += 0.05 * np.random.randn()
        unique_x2 += 0.05 * np.random.randn()
        synergy += 0.05 * np.random.randn()
        
        # Ensure non-negative values
        redundancy = max(0, redundancy)
        unique_x1 = max(0, unique_x1)
        unique_x2 = max(0, unique_x2)
        synergy = max(0, synergy)
        
        # Total information
        total_di = redundancy + unique_x1 + unique_x2 + synergy
        
        # Synergy ratio
        synergy_ratio = synergy / total_di if total_di > 0 else 0
        
        # Store results
        window_results['window_start'].append(window_start)
        window_results['window_end'].append(window_end)
        window_results['redundancy'].append(redundancy)
        window_results['unique_x1'].append(unique_x1)
        window_results['unique_x2'].append(unique_x2)
        window_results['synergy'].append(synergy)
        window_results['total_di'].append(total_di)
        window_results['synergy_ratio'].append(synergy_ratio)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(window_results)
    
    # Identify high synergy segments
    high_synergy_mask = results_df['synergy'] > min_synergy_threshold
    high_synergy_segments = results_df[high_synergy_mask]
    
    # Convert to list of tuples
    high_synergy_list = [
        (row['window_start'], row['window_end'], row['synergy']) 
        for _, row in high_synergy_segments.iterrows()
    ]
    
    return results_df, high_synergy_list

def simulate_model_performance(X1, X2, synergy_segments, non_synergy_segments):
    """
    Simulate model performance results for visualization.
    In a real implementation, this would train and evaluate actual models.
    """
    # Create mock performance results showing better combined performance in synergy regions
    syn_results = {
        'X1': {'accuracy': 0.65, 'f1_score': 0.63},
        'X2': {'accuracy': 0.67, 'f1_score': 0.66},
        'X1X2': {'accuracy': 0.85, 'f1_score': 0.84}  # Much better combined
    }
    
    non_syn_results = {
        'X1': {'accuracy': 0.73, 'f1_score': 0.72},
        'X2': {'accuracy': 0.75, 'f1_score': 0.73},
        'X1X2': {'accuracy': 0.79, 'f1_score': 0.77}  # Only slightly better
    }
    
    return syn_results

def visualize_results(results_df, high_synergy_segments, true_segments, model_results=None, save_path=None):
    """Visualize PID components, detected synergy segments, and model validation."""
    # Create figures directory if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Time points for x-axis
    time_points = np.array(results_df['window_start']) + (np.array(results_df['window_end']) - np.array(results_df['window_start'])) / 2
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Determine number of subplots
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
    
    # Highlight true high synergy regions
    for start, end in true_segments:
        plt.axvspan(start, end, color='green', alpha=0.2)
            
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Synergy (bits)', fontsize=12)
    plt.title('Detected High Synergy Segments', fontsize=14)
    plt.legend(['Synergy', 'Detected Segments', 'True Segments'])
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Model validation results if available
    if model_results:
        plt.subplot(n_plots, 1, 3)
        
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
        plt.title('Model Performance in High Synergy Regions', fontsize=14)
        plt.xticks(x_pos, modalities)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center')
        for i, v in enumerate(f1_scores):
            plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_overview.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}_overview.png")
    else:
        plt.show()

def main():
    """Main function to demonstrate multimodal synergy detection concept."""
    try:
        print("\n=== Multimodal Synergy Detection (Conceptual Demonstration) ===\n")
        
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
        
        # 2. Simulate PID analysis to detect segments with high synergy
        print("\nAnalyzing multimodal synergy (simulated results)...")
        results_df, detected_segments = simulate_pid_results(
            X1, X2, Y,
            window_size=30,
            stride=15,
            min_synergy_threshold=0.1
        )
        
        print(f"\nDetected {len(detected_segments)} segments with high synergy:")
        for i, (start, end, synergy) in enumerate(detected_segments[:5]):
            print(f"  Segment {i+1}: Time {start}-{end}, Synergy: {synergy:.4f}")
        
        # 3. Simulate model validation results
        print("\nValidating insights with models (simulated results)...")
        # Define non-synergy segments as complement of detected segments
        all_indices = set(range(len(X1)))
        synergy_indices = set()
        for start, end, _ in detected_segments:
            synergy_indices.update(range(start, end))
        non_synergy_indices = all_indices - synergy_indices
        
        model_results = simulate_model_performance(
            X1, X2, 
            synergy_segments=detected_segments,
            non_synergy_segments=non_synergy_indices
        )
        
        print("\nModel performance in high synergy segments:")
        for modality, metrics in model_results.items():
            print(f"  {modality}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
        
        # 4. Visualize results
        print("\nVisualizing results...")
        visualize_results(
            results_df, 
            detected_segments, 
            true_segments=true_synergy_segments,
            model_results=model_results,
            save_path='results/multimodal_synergy_concept'
        )
        
        print("\nDemonstration complete!")
        print("\nNote: This is a conceptual demonstration with simulated PID results.")
        print("In a full implementation, the temporal_pid function would be used to")
        print("calculate actual information-theoretic measures for each time window.")
        
    except Exception as e:
        print(f"\nError in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 