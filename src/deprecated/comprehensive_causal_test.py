import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score
from scipy import stats

# Make sure we can import from src directory
sys.path.append('src')
try:
    from temporal_pid import temporal_pid, multi_lag_analysis, plot_multi_lag_results
except ImportError:
    print("Error importing temporal_pid. Make sure the src directory exists and contains temporal_pid.py.")
    sys.exit(1)

#----------------------------------------------------------------------------------
# Data Generation Functions for Different Causal Structures
#----------------------------------------------------------------------------------

def generate_chain_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a chain structure: X → Z → Y
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'x_to_z': 0.8, 'z_to_y': 0.8}
    
    X = np.random.randn(n_samples)
    e_z = np.random.randn(n_samples) * noise_level
    e_y = np.random.randn(n_samples) * noise_level
    
    Z = coefficients['x_to_z'] * X + e_z
    Y = coefficients['z_to_y'] * Z + e_y
    
    return X, Z, Y

def generate_fork_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a fork/common cause structure: X ← Z → Y
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'z_to_x': 0.8, 'z_to_y': 0.8}
    
    Z = np.random.randn(n_samples)
    e_x = np.random.randn(n_samples) * noise_level
    e_y = np.random.randn(n_samples) * noise_level
    
    X = coefficients['z_to_x'] * Z + e_x
    Y = coefficients['z_to_y'] * Z + e_y
    
    return X, Z, Y

def generate_v_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a v-structure/collider: X → Z ← Y
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'x_to_z': 0.8, 'y_to_z': 0.8}
    
    X = np.random.randn(n_samples)
    Y = np.random.randn(n_samples)
    e_z = np.random.randn(n_samples) * noise_level
    
    Z = coefficients['x_to_z'] * X + coefficients['y_to_z'] * Y + e_z
    
    return X, Z, Y

#----------------------------------------------------------------------------------
# Analysis Functions
#----------------------------------------------------------------------------------

def compare_pid_across_structures(results_dict, save_path=None):
    """
    Compare PID results across different causal structures.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Set up colors for different structures
    colors = {'Chain': 'blue', 'Fork': 'green', 'V-structure': 'red'}
    
    # Plot each PID component across structures
    components = ['redundancy', 'unique_x1', 'unique_x2', 'synergy']
    component_names = ['Redundancy', 'Unique X', 'Unique Y', 'Synergy']
    
    for i, (component, name) in enumerate(zip(components, component_names)):
        ax = axes[i]
        
        for structure, results in results_dict.items():
            ax.plot(results['lag'], results[component], '-o', 
                    label=structure, color=colors.get(structure, 'black'), linewidth=2)
            
        ax.set_xlabel('Time Lag', fontsize=14)
        ax.set_ylabel('Information (bits)', fontsize=14)
        ax.set_title(f'{name} Across Causal Structures', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    else:
        plt.show()

def run_statistical_tests(results_dict):
    """
    Run statistical tests to determine if PID can significantly distinguish between structures.
    """
    print("\n=== Statistical Analysis ===")
    structures = list(results_dict.keys())
    components = ['redundancy', 'unique_x1', 'unique_x2', 'synergy']
    component_names = ['Redundancy', 'Unique X', 'Unique Y', 'Synergy']
    
    stats_results = {}
    
    # Compare each pair of structures
    for i, comp1 in enumerate(components):
        print(f"\nComponent: {component_names[i]}")
        stats_results[comp1] = {}
        
        for j in range(len(structures)):
            for k in range(j+1, len(structures)):
                struct1 = structures[j]
                struct2 = structures[k]
                
                # Get the values for this component from both structures
                values1 = results_dict[struct1][comp1]
                values2 = results_dict[struct2][comp1]
                
                # Run t-test to check for significant differences
                t_stat, p_val = stats.ttest_ind(values1, values2)
                
                # Store and print results
                comparison = f"{struct1} vs {struct2}"
                stats_results[comp1][comparison] = p_val
                
                print(f"  {comparison}: p-value = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")
    
    return stats_results

def generate_signature_plot(results_dict, save_path=None):
    """
    Generate a "signature plot" of RUS pattern for each causal structure.
    """
    fig, axes = plt.subplots(1, len(results_dict), figsize=(6*len(results_dict), 8))
    if len(results_dict) == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'red', 'magenta']
    components = ['redundancy', 'unique_x1', 'unique_x2', 'synergy']
    component_names = ['Redundancy', 'Unique X', 'Unique Y', 'Synergy']
    
    # For each structure, create a radar/spider plot of the average RUS values
    for i, (structure, results) in enumerate(results_dict.items()):
        ax = axes[i]
        
        # Calculate mean values for each component
        means = [np.mean(results[comp]) for comp in components]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(components), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        means += means[:1]  # Close the loop
        
        ax.plot(angles, means, 'o-', linewidth=2, color='blue')
        ax.fill(angles, means, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(component_names)
        ax.set_title(f'{structure} PID Signature', fontsize=16)
        
        # Add grid lines and labels
        ax.set_ylim(0, max(max([max(results[comp]) for comp in components]) * 1.2, 0.01))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Signature plot saved to {save_path}")
    else:
        plt.show()

def run_multiple_trials(structure_generator, n_trials=10, n_samples=1000, max_lag=3, bins=8, seed=None):
    """
    Run multiple trials for a given causal structure to get more robust statistics.
    """
    # Initialize aggregated results
    agg_results = {
        'redundancy': np.zeros((n_trials, max_lag)),
        'unique_x1': np.zeros((n_trials, max_lag)),
        'unique_x2': np.zeros((n_trials, max_lag)),
        'synergy': np.zeros((n_trials, max_lag)),
        'total_di': np.zeros((n_trials, max_lag))
    }
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...")
        
        # Generate data with different random seed for each trial
        trial_seed = seed + trial if seed is not None else None
        X, Z, Y = structure_generator(n_samples=n_samples, noise_level=0.1, seed=trial_seed)
        
        # Perform PID analysis
        results = multi_lag_analysis(X, Y, Z, max_lag=max_lag, bins=bins)
        
        # Store results
        for component in ['redundancy', 'unique_x1', 'unique_x2', 'synergy', 'total_di']:
            agg_results[component][trial, :] = results[component]
    
    # Calculate mean and standard deviation
    mean_results = {
        'lag': list(range(1, max_lag+1)),
        'redundancy': np.mean(agg_results['redundancy'], axis=0),
        'unique_x1': np.mean(agg_results['unique_x1'], axis=0),
        'unique_x2': np.mean(agg_results['unique_x2'], axis=0),
        'synergy': np.mean(agg_results['synergy'], axis=0),
        'total_di': np.mean(agg_results['total_di'], axis=0)
    }
    
    std_results = {
        'redundancy': np.std(agg_results['redundancy'], axis=0),
        'unique_x1': np.std(agg_results['unique_x1'], axis=0),
        'unique_x2': np.std(agg_results['unique_x2'], axis=0),
        'synergy': np.std(agg_results['synergy'], axis=0),
        'total_di': np.std(agg_results['total_di'], axis=0)
    }
    
    return mean_results, std_results, agg_results

#----------------------------------------------------------------------------------
# Main Analysis
#----------------------------------------------------------------------------------

def main():
    print("Starting comprehensive causal structure test script...")
    
    # Set up parameters
    n_samples = 1000
    n_trials = 5  # Number of trials for robust statistics
    noise_level = 0.1
    max_lag = 3
    bins = 8
    seed = 42
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("Running multiple trials for each causal structure...")
    
    # Chain structure
    print("\nAnalyzing Chain Structure (X → Z → Y)...")
    chain_mean, chain_std, chain_agg = run_multiple_trials(
        generate_chain_structure, n_trials=n_trials, n_samples=n_samples, 
        max_lag=max_lag, bins=bins, seed=seed)
    
    # Fork structure
    print("\nAnalyzing Fork Structure (X ← Z → Y)...")
    fork_mean, fork_std, fork_agg = run_multiple_trials(
        generate_fork_structure, n_trials=n_trials, n_samples=n_samples, 
        max_lag=max_lag, bins=bins, seed=seed)
    
    # V-structure
    print("\nAnalyzing V-Structure (X → Z ← Y)...")
    v_mean, v_std, v_agg = run_multiple_trials(
        generate_v_structure, n_trials=n_trials, n_samples=n_samples, 
        max_lag=max_lag, bins=bins, seed=seed)
    
    # Collect results
    all_results = {
        'Chain': chain_mean,
        'Fork': fork_mean,
        'V-structure': v_mean
    }
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    compare_pid_across_structures(all_results, 
                               save_path='results/comprehensive_causal_comparison.png')
    
    # Generate signature plots
    generate_signature_plot(all_results, 
                         save_path='results/comprehensive_causal_signatures.png')
    
    # Run statistical tests
    stats_results = run_statistical_tests(all_results)
    
    # Plot individual components with error bars
    print("\nGenerating error bar plots...")
    components = ['redundancy', 'unique_x1', 'unique_x2', 'synergy']
    component_names = ['Redundancy', 'Unique X', 'Unique Y', 'Synergy']
    
    for i, (component, name) in enumerate(zip(components, component_names)):
        plt.figure(figsize=(10, 6))
        
        x = list(range(1, max_lag+1))
        
        # Plot with error bars
        plt.errorbar(x, chain_mean[component], yerr=chain_std[component], 
                    fmt='o-', label='Chain', color='blue', capsize=4)
        plt.errorbar(x, fork_mean[component], yerr=fork_std[component], 
                    fmt='s-', label='Fork', color='green', capsize=4)
        plt.errorbar(x, v_mean[component], yerr=v_std[component], 
                    fmt='^-', label='V-structure', color='red', capsize=4)
        
        plt.xlabel('Time Lag', fontsize=14)
        plt.ylabel('Information (bits)', fontsize=14)
        plt.title(f'{name} Across Causal Structures (Mean ± Std)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.savefig(f'results/comprehensive_{component}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  {name} comparison saved.")
    
    # Calculate and print significance matrix
    print("\n=== Summary of Statistical Significance ===")
    for component, name in zip(components, component_names):
        print(f"\n{name}:")
        structures = list(all_results.keys())
        print("  " + "  ".join([f"{s:<12}" for s in structures]))
        
        for i, s1 in enumerate(structures):
            row = [s1]
            for j, s2 in enumerate(structures):
                if i == j:
                    row.append("---")
                else:
                    if i < j:
                        comparison = f"{s1} vs {s2}"
                        p_val = stats_results[component].get(comparison, 1.0)
                    else:
                        comparison = f"{s2} vs {s1}"
                        p_val = stats_results[component].get(comparison, 1.0)
                    
                    if p_val < 0.01:
                        row.append("***")  # Highly significant
                    elif p_val < 0.05:
                        row.append("**")   # Significant
                    elif p_val < 0.1:
                        row.append("*")    # Marginally significant
                    else:
                        row.append("ns")   # Not significant
            
            print("  " + "  ".join([f"{r:<12}" for r in row]))
    
    print("\nAnalysis complete. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main() 