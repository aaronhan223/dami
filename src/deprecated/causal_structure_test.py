import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score
import cvxpy as cp
import seaborn as sns
from scipy import stats

# Make sure we can import from src directory
sys.path.append('src')
try:
    from temporal_pid import (temporal_pid, create_probability_distribution, 
                             solve_Q_temporal, CoI_temporal, UI_temporal, 
                             CI_temporal, MI, multi_lag_analysis, 
                             plot_multi_lag_results)
except ImportError:
    print("Error importing temporal_pid. Make sure the src directory exists and contains temporal_pid.py.")
    sys.exit(1)

#----------------------------------------------------------------------------------
# Data Generation Functions for Different Causal Structures
#----------------------------------------------------------------------------------

def generate_chain_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a chain structure: X → Z → Y
    
    In a chain structure, X influences Z, which in turn influences Y.
    There is no direct path from X to Y.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    coefficients : dict, optional
        Dictionary containing the causal coefficients:
        'x_to_z': Strength of X's influence on Z
        'z_to_y': Strength of Z's influence on Y
    noise_level : float, default=0.1
        Magnitude of the random noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X, Z, Y : numpy.ndarray
        Generated time series data
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'x_to_z': 0.8, 'z_to_y': 0.8}
    
    # Generate exogenous variables
    X = np.random.randn(n_samples)
    e_z = np.random.randn(n_samples) * noise_level
    e_y = np.random.randn(n_samples) * noise_level
    
    # Generate the causal structure
    Z = coefficients['x_to_z'] * X + e_z
    Y = coefficients['z_to_y'] * Z + e_y
    
    return X, Z, Y

def generate_fork_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a fork/common cause structure: X ← Z → Y
    
    In a fork structure, Z is a common cause of both X and Y.
    X and Y are conditionally independent given Z.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    coefficients : dict, optional
        Dictionary containing the causal coefficients:
        'z_to_x': Strength of Z's influence on X
        'z_to_y': Strength of Z's influence on Y
    noise_level : float, default=0.1
        Magnitude of the random noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X, Z, Y : numpy.ndarray
        Generated time series data
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'z_to_x': 0.8, 'z_to_y': 0.8}
    
    # Generate exogenous variables
    Z = np.random.randn(n_samples)
    e_x = np.random.randn(n_samples) * noise_level
    e_y = np.random.randn(n_samples) * noise_level
    
    # Generate the causal structure
    X = coefficients['z_to_x'] * Z + e_x
    Y = coefficients['z_to_y'] * Z + e_y
    
    return X, Z, Y

def generate_v_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a v-structure/collider: X → Z ← Y
    
    In a v-structure, both X and Y independently influence Z.
    X and Y are marginally independent but conditionally dependent given Z.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    coefficients : dict, optional
        Dictionary containing the causal coefficients:
        'x_to_z': Strength of X's influence on Z
        'y_to_z': Strength of Y's influence on Z
    noise_level : float, default=0.1
        Magnitude of the random noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X, Z, Y : numpy.ndarray
        Generated time series data
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'x_to_z': 0.8, 'y_to_z': 0.8}
    
    # Generate exogenous variables
    X = np.random.randn(n_samples)
    Y = np.random.randn(n_samples)
    e_z = np.random.randn(n_samples) * noise_level
    
    # Generate the causal structure
    Z = coefficients['x_to_z'] * X + coefficients['y_to_z'] * Y + e_z
    
    return X, Z, Y

#----------------------------------------------------------------------------------
# Analysis Functions
#----------------------------------------------------------------------------------

def run_pid_analysis(X, Z, Y, description, max_lag=3, bins=10):
    """
    Run PID analysis on the given causal structure and print/plot results.
    
    Parameters:
    -----------
    X, Z, Y : numpy.ndarray
        Time series data from a specific causal structure
    description : str
        Description of the causal structure
    max_lag : int, default=3
        Maximum lag to consider in the analysis
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    pid_results : dict
        Dictionary containing the PID analysis results
    """
    print(f"\n=== PID Analysis for {description} ===")
    
    # For the PID analysis, we're always analyzing the influence of X and Y on Z
    # This is appropriate for all structures: causal influence in v-structure,
    # potential redundancy in fork, and indirect paths in chain
    
    # Run multi-lag PID analysis
    results = multi_lag_analysis(X, Y, Z, max_lag=max_lag, bins=bins)
    
    # Print summary statistics
    print(f"Average Redundancy: {np.mean(results['redundancy']):.4f}")
    print(f"Average Unique X: {np.mean(results['unique_x1']):.4f}")
    print(f"Average Unique Y: {np.mean(results['unique_x2']):.4f}")
    print(f"Average Synergy: {np.mean(results['synergy']):.4f}")
    
    # Return results for further analysis
    return results

def compare_pid_across_structures(results_dict, save_path=None):
    """
    Compare PID results across different causal structures.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping structure names to their PID results
    save_path : str, optional
        Path to save the figure
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
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping structure names to their PID results
        
    Returns:
    --------
    stats_results : dict
        Dictionary containing p-values for different comparisons
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
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping structure names to their PID results
    save_path : str, optional
        Path to save the figure
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

#----------------------------------------------------------------------------------
# Main Analysis
#----------------------------------------------------------------------------------

def main():
    # Set up parameters
    n_samples = 2000
    noise_level = 0.1
    max_lag = 5
    bins = 10
    seed = 42
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("Generating data for different causal structures...")
    
    # Generate data for each causal structure
    X_chain, Z_chain, Y_chain = generate_chain_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    
    X_fork, Z_fork, Y_fork = generate_fork_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    
    X_v, Z_v, Y_v = generate_v_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    
    # Run PID analysis on each structure
    results_chain = run_pid_analysis(X_chain, Z_chain, Y_chain, "Chain Structure (X → Z → Y)", max_lag, bins)
    results_fork = run_pid_analysis(X_fork, Z_fork, Y_fork, "Fork Structure (X ← Z → Y)", max_lag, bins)
    results_v = run_pid_analysis(X_v, Z_v, Y_v, "V-Structure (X → Z ← Y)", max_lag, bins)
    
    # Collect results
    all_results = {
        'Chain': results_chain,
        'Fork': results_fork,
        'V-structure': results_v
    }
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    compare_pid_across_structures(all_results, save_path='results/causal_structures_comparison.png')
    
    # Generate signature plots
    generate_signature_plot(all_results, save_path='results/causal_structures_signatures.png')
    
    # Run statistical tests
    run_statistical_tests(all_results)
    
    # Individual lag analyses and plots
    for structure, results in all_results.items():
        plot_multi_lag_results(results, save_path=f'results/{structure.lower().replace("-", "_")}_pid_results.png')
    
    print("\nAnalysis complete. Results and plots saved to the 'results' directory.")

if __name__ == "__main__":
    main() 