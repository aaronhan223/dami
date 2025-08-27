"""
RUS Estimation Plotting Functions

This module contains plotting functions for RUS (Redundancy, Uniqueness, Synergy) 
estimation analysis, particularly for comparing different methods and parameter settings.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_experiment_results(results_dir: str, method_filter: str = None) -> Dict[int, Dict]:
    """
    Load experiment results from a directory containing multiple n_components experiments.
    
    Args:
        results_dir: Directory containing experiment subdirectories
        method_filter: Filter results by MI method ('neural', 'hybrid', 'knn', or None for all)
        
    Returns:
        Dictionary mapping n_components values to experiment results
    """
    results = {}
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results
    
    # Find all experiment subdirectories
    if method_filter:
        # Look for method-specific patterns
        if method_filter == 'knn':
            pattern = os.path.join(results_dir, f"rus_estimate_gpu2_{method_filter}_*")
        else:
            pattern = os.path.join(results_dir, f"rus_estimate_gpu2_{method_filter}_components*")
    else:
        # Look for any experiment directories with components (backward compatibility)
        pattern = os.path.join(results_dir, "rus_estimate_gpu2_*components*")
    
    exp_dirs = glob.glob(pattern)
    
    for exp_dir in exp_dirs:
        # Extract n_components from directory name
        dir_name = os.path.basename(exp_dir)
        try:
            # Extract method and n_components from experiment name
            if method_filter == 'knn' and f"rus_estimate_gpu2_{method_filter}" in dir_name:
                # KNN method doesn't use n_components
                n_components = 'knn'
            elif "components" in dir_name:
                components_str = dir_name.split("components")[1].split("_")[0]
                n_components = int(components_str)
            else:
                continue
        except (ValueError, IndexError):
            continue
        
        # Find pickle file with results
        pickle_files = glob.glob(os.path.join(exp_dir, "*_temporal_tracking_results.pkl"))
        if not pickle_files:
            continue
        
        # Load the most recent results file
        pickle_file = max(pickle_files, key=os.path.getmtime)
        
        try:
            with open(pickle_file, 'rb') as f:
                experiment_data = pickle.load(f)
                results[n_components] = experiment_data
                print(f"Loaded results for n_components={n_components} (method: {method_filter or 'any'}) from {pickle_file}")
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            continue
    
    return results

def load_batch_comparison_data(batch_file_path: str) -> Optional[List[Dict]]:
    """Load batch estimator results for comparison."""
    if not os.path.exists(batch_file_path):
        print(f"Batch comparison file not found: {batch_file_path}")
        return None
    
    try:
        data = np.load(batch_file_path, allow_pickle=True)
        print(f"Loaded batch comparison data with {len(data)} modality pairs")
        return data
    except Exception as e:
        print(f"Error loading batch comparison data: {e}")
        return None

def compute_accuracy_metrics(tracking_results: List[Dict], batch_results: List[Dict]) -> Dict:
    """
    Compute accuracy metrics comparing tracking results with batch results.
    Returns mean absolute errors averaged over all time lags.
    """
    if not batch_results:
        return {}
    
    # Find matching modality pairs between tracking and batch results
    batch_dict = {}
    for batch_data in batch_results:
        pair = batch_data.get('feature_pair', (None, None))
        if pair != (None, None):
            batch_dict[pair] = batch_data
    
    accuracy_metrics = {
        'mean_abs_errors': {'R': [], 'U1': [], 'U2': [], 'S': []},
        'modality_pairs': []
    }
    
    for comparison in tracking_results:
        modality_pair = comparison.get('modality_pair', (None, None))
        if modality_pair == (None, None):
            continue
        
        # Try both orientations of the pair
        batch_data = batch_dict.get(modality_pair) or batch_dict.get(modality_pair[::-1])
        if not batch_data:
            continue
        
        accuracy_metrics['modality_pairs'].append(modality_pair)
        
        # Get tracking results and batch results
        tracking_lag_results = comparison['tracking_results']
        batch_lag_results = batch_data['lag_results']
        
        # Create lookup for batch results by lag
        batch_by_lag = {result['lag']: result for result in batch_lag_results}
        
        # Compute errors for each lag
        pair_errors = {'R': [], 'U1': [], 'U2': [], 'S': []}
        
        for tracking_result in tracking_lag_results:
            lag = tracking_result['lag']
            if lag not in batch_by_lag:
                continue
            
            batch_result = batch_by_lag[lag]
            
            # Compute absolute errors
            pair_errors['R'].append(abs(tracking_result['R_value'] - batch_result['R_value']))
            pair_errors['U1'].append(abs(tracking_result['U1_value'] - batch_result['U1_value']))
            pair_errors['U2'].append(abs(tracking_result['U2_value'] - batch_result['U2_value']))
            pair_errors['S'].append(abs(tracking_result['S_value'] - batch_result['S_value']))
        
        # Average errors across all lags for this pair
        for component in ['R', 'U1', 'U2', 'S']:
            if pair_errors[component]:
                mean_error = np.mean(pair_errors[component])
                accuracy_metrics['mean_abs_errors'][component].append(mean_error)
    
    # Average across all modality pairs
    for component in ['R', 'U1', 'U2', 'S']:
        if accuracy_metrics['mean_abs_errors'][component]:
            accuracy_metrics['mean_abs_errors'][component] = np.mean(
                accuracy_metrics['mean_abs_errors'][component]
            )
        else:
            accuracy_metrics['mean_abs_errors'][component] = np.nan
    
    return accuracy_metrics

def plot_n_components_comparison(results_dir: str, batch_file_path: str, output_dir: str = None, method_filter: str = None):
    """
    Plot comparison of RUS estimation accuracy vs n_components values.
    
    Args:
        results_dir: Directory containing experiment results for different n_components
        batch_file_path: Path to batch estimator results file
        output_dir: Directory to save plots (defaults to results_dir)
        method_filter: Filter results by MI method ('neural', 'hybrid', or None for all with components)
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment results with method filtering
    experiment_results = load_experiment_results(results_dir, method_filter=method_filter)
    if not experiment_results:
        print(f"No experiment results found for method: {method_filter or 'any'}!")
        return
    
    # Load batch comparison data
    batch_results = load_batch_comparison_data(batch_file_path)
    if not batch_results:
        print("No batch comparison data available!")
        return
    
    print(f"Found results for n_components values: {sorted(experiment_results.keys())}")
    
    # Compute accuracy metrics for each n_components value
    n_components_values = []
    accuracy_data = {'R': [], 'U1': [], 'U2': [], 'S': []}
    
    for n_comp in sorted(experiment_results.keys()):
        tracking_results = experiment_results[n_comp]
        accuracy_metrics = compute_accuracy_metrics(tracking_results, batch_results)
        
        if accuracy_metrics and 'mean_abs_errors' in accuracy_metrics:
            n_components_values.append(n_comp)
            for component in ['R', 'U1', 'U2', 'S']:
                accuracy_data[component].append(accuracy_metrics['mean_abs_errors'][component])
    
    if not n_components_values:
        print("No valid accuracy data computed!")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors and markers for each RUS component
    colors = {
        'R': '#1f77b4',    # Blue - Redundancy
        'U1': '#ff7f0e',   # Orange - Unique X1
        'U2': '#2ca02c',   # Green - Unique X2
        'S': '#d62728'     # Red - Synergy
    }
    
    markers = {
        'R': 'o',
        'U1': 's', 
        'U2': '^',
        'S': 'd'
    }
    
    labels = {
        'R': 'Redundancy',
        'U1': 'Unique X1',
        'U2': 'Unique X2', 
        'S': 'Synergy'
    }
    
    # Plot accuracy curves for each component
    for component in ['R', 'U1', 'U2', 'S']:
        valid_mask = ~np.isnan(accuracy_data[component])
        if np.any(valid_mask):
            x_vals = np.array(n_components_values)[valid_mask]
            y_vals = np.array(accuracy_data[component])[valid_mask]
            
            ax.plot(x_vals, y_vals, 
                   marker=markers[component], 
                   color=colors[component],
                   label=f'{labels[component]} MAE',
                   linewidth=2, 
                   markersize=8,
                   markerfacecolor=colors[component],
                   markeredgecolor='white',
                   markeredgewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Number of Components (n_components)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (bits)', fontsize=14, fontweight='bold')
    
    # Create title based on method filter
    method_str = f" ({method_filter.upper()} Method)" if method_filter else ""
    title = f'RUS Estimation Accuracy vs Number of Components{method_str}\n(PAMAP2 Subject 3, Averaged over All Time Lags)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, framealpha=0.9, loc='best')
    
    # Set x-ticks to show all n_components values
    ax.set_xticks(n_components_values)
    ax.set_xlim(min(n_components_values) - 1, max(n_components_values) + 1)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Add minor grid
    ax.grid(True, which='minor', alpha=0.1)
    ax.minorticks_on()
    
    plt.tight_layout()
    
    # Create filename based on method filter
    method_suffix = f"_{method_filter}" if method_filter else ""
    plot_filename = f'pamap_n_components_accuracy_comparison{method_suffix}.png'
    pdf_filename = f'pamap_n_components_accuracy_comparison{method_suffix}.pdf'
    
    # Save the plot
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"N-components comparison plot saved to: {plot_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, pdf_filename)
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"N-components comparison plot (PDF) saved to: {pdf_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("N-COMPONENTS ACCURACY COMPARISON SUMMARY")
    print("="*80)
    
    print(f"Tested n_components values: {n_components_values}")
    print(f"Number of modality pairs analyzed: {len(accuracy_data['R']) if accuracy_data['R'] else 0}")
    
    for component in ['R', 'U1', 'U2', 'S']:
        if accuracy_data[component] and not all(np.isnan(accuracy_data[component])):
            values = np.array(accuracy_data[component])
            values = values[~np.isnan(values)]
            
            best_idx = np.argmin(values)
            best_n_comp = n_components_values[best_idx]
            best_error = values[best_idx]
            
            print(f"\n{labels[component]} (MAE):")
            print(f"  Best n_components: {best_n_comp} (MAE: {best_error:.6f} bits)")
            print(f"  Range: {np.min(values):.6f} - {np.max(values):.6f} bits")
            print(f"  Mean: {np.mean(values):.6f}  {np.std(values):.6f} bits")
    
    # Save numerical results
    results_data = {
        'n_components_values': n_components_values,
        'accuracy_data': accuracy_data,
        'experiment_info': {
            'results_dir': results_dir,
            'batch_file': batch_file_path,
            'num_experiments': len(experiment_results),
            'description': 'Mean absolute errors averaged over all time lags and modality pairs'
        }
    }
    
    results_file = os.path.join(output_dir, 'n_components_accuracy_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\nNumerical results saved to: {results_file}")

def plot_individual_n_components_results(results_dir: str, n_components_list: List[int] = None, 
                                       output_dir: str = None):
    """
    Plot individual RUS estimation results for specific n_components values.
    
    Args:
        results_dir: Directory containing experiment results
        n_components_list: List of n_components values to plot (defaults to all available)
        output_dir: Directory to save plots (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment results
    experiment_results = load_experiment_results(results_dir)
    if not experiment_results:
        print("No experiment results found!")
        return
    
    if n_components_list is None:
        n_components_list = sorted(experiment_results.keys())
    
    # Filter to available results
    available_n_comp = [n for n in n_components_list if n in experiment_results]
    
    if not available_n_comp:
        print(f"No results found for requested n_components values: {n_components_list}")
        return
    
    print(f"Plotting results for n_components: {available_n_comp}")
    
    # Create subplot grid
    n_plots = len(available_n_comp)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for RUS components
    colors = {
        'R': '#1f77b4',    # Blue - Redundancy
        'U1': '#ff7f0e',   # Orange - Unique X1
        'U2': '#2ca02c',   # Green - Unique X2
        'S': '#d62728'     # Red - Synergy
    }
    
    labels = {
        'R': 'Redundancy',
        'U1': 'Unique X1',
        'U2': 'Unique X2',
        'S': 'Synergy'
    }
    
    for idx, n_comp in enumerate(available_n_comp):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        tracking_results = experiment_results[n_comp]
        
        # Average results across all modality pairs
        all_lags = set()
        for comparison in tracking_results:
            for result in comparison['tracking_results']:
                all_lags.add(result['lag'])
        
        lags = sorted(all_lags)
        avg_values = {'R': [], 'U1': [], 'U2': [], 'S': []}
        
        for lag in lags:
            lag_values = {'R': [], 'U1': [], 'U2': [], 'S': []}
            
            for comparison in tracking_results:
                for result in comparison['tracking_results']:
                    if result['lag'] == lag:
                        lag_values['R'].append(result['R_value'])
                        lag_values['U1'].append(result['U1_value'])
                        lag_values['U2'].append(result['U2_value'])
                        lag_values['S'].append(result['S_value'])
            
            for component in ['R', 'U1', 'U2', 'S']:
                if lag_values[component]:
                    avg_values[component].append(np.mean(lag_values[component]))
                else:
                    avg_values[component].append(0)
        
        # Plot each component
        for component in ['R', 'U1', 'U2', 'S']:
            ax.plot(lags, avg_values[component], 'o-', 
                   color=colors[component], label=labels[component],
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Time Lag', fontsize=10)
        ax.set_ylabel('Information (bits)', fontsize=10)
        ax.set_title(f'n_components = {n_comp}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_xticks(lags)
    
    # Remove empty subplots
    if n_plots < n_rows * n_cols:
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
    
    plt.tight_layout()
    plt.suptitle('RUS Estimation Results by n_components\n(PAMAP2 Subject 3, Averaged over Modality Pairs)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'pamap_individual_n_components_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Individual n_components results plot saved to: {plot_path}")
    
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Example paths (update these for actual usage)
    results_dir = "../results/pamap_n_components_sweep" 
    batch_file = "../results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy"
    
    print("Testing RUS estimation plotting functions...")
    
    # Test the main comparison plot
    try:
        plot_n_components_comparison(results_dir, batch_file)
        print(" N-components comparison plot completed successfully")
    except Exception as e:
        print(f" Error in n_components comparison plot: {e}")
    
    # Test individual results plot
    try:
        plot_individual_n_components_results(results_dir, n_components_list=[8, 16, 24])
        print(" Individual n_components results plot completed successfully")
    except Exception as e:
        print(f" Error in individual results plot: {e}")