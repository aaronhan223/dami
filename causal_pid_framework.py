import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score
from itertools import permutations
import pdb
# Make sure we can import from src directory
sys.path.append('src')
try:
    from temporal_pid import temporal_pid, create_probability_distribution, estimate_transfer_entropy
except ImportError:
    print("Error importing temporal_pid. Make sure the src directory exists and contains temporal_pid.py.")
    sys.exit(1)

#----------------------------------------------------------------------------------
# Multi-Lag Directed Information PID Framework
#----------------------------------------------------------------------------------

def estimate_multivariate_directed_information(X, Y, Z, max_lag=5, bins=10, method='binned'):
    """
    Estimate directed information from multiple sources to target with varying lags.
    
    This function computes directed information (via transfer entropy) between multiple
    variables across different time lags. It's designed to detect causal relationships
    that may only be visible at specific time scales.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data for the three variables
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
    method : str, default='binned'
        Method for estimation: 'binned' or 'ksg'
        
    Returns:
    --------
    dict
        Dictionary containing directed information measures for various relationships
        at different lags
    """
    # Initialize result dictionary
    results = {
        'lags': list(range(1, max_lag + 1)),
        'di_x_to_z': np.zeros(max_lag),
        'di_z_to_x': np.zeros(max_lag),
        'di_y_to_z': np.zeros(max_lag),
        'di_z_to_y': np.zeros(max_lag),
        'di_x_to_y': np.zeros(max_lag),
        'di_y_to_x': np.zeros(max_lag),
        'causal_idx_xz': np.zeros(max_lag),
        'causal_idx_yz': np.zeros(max_lag),
        'causal_idx_xy': np.zeros(max_lag)
    }
    
    # Calculate directed information (transfer entropy) for each lag
    for lag_idx, lag in enumerate(range(1, max_lag + 1)):
        # Direct causal effects (X→Z, Y→Z)
        results['di_x_to_z'][lag_idx] = estimate_transfer_entropy(X, Z, lag=lag, bins=bins, method=method)
        results['di_z_to_x'][lag_idx] = estimate_transfer_entropy(Z, X, lag=lag, bins=bins, method=method)
        results['di_y_to_z'][lag_idx] = estimate_transfer_entropy(Y, Z, lag=lag, bins=bins, method=method)
        results['di_z_to_y'][lag_idx] = estimate_transfer_entropy(Z, Y, lag=lag, bins=bins, method=method)
        results['di_x_to_y'][lag_idx] = estimate_transfer_entropy(X, Y, lag=lag, bins=bins, method=method)
        results['di_y_to_x'][lag_idx] = estimate_transfer_entropy(Y, X, lag=lag, bins=bins, method=method)
        
        # Calculate causal indices (difference between forward and backward TE)
        # Positive values indicate X→Z is stronger than Z→X (suggesting causal relationship)
        results['causal_idx_xz'][lag_idx] = results['di_x_to_z'][lag_idx] - results['di_z_to_x'][lag_idx]
        results['causal_idx_yz'][lag_idx] = results['di_y_to_z'][lag_idx] - results['di_z_to_y'][lag_idx]
        results['causal_idx_xy'][lag_idx] = results['di_x_to_y'][lag_idx] - results['di_y_to_x'][lag_idx]
    
    return results

def multi_lag_partial_directed_information(X, Y, Z, max_lag=5, bins=10):
    """
    Compute a full multi-lag PID analysis using directed information.
    
    This function extends standard PID by analyzing information flow at multiple lags,
    revealing time-delayed causal relationships that might be invisible to standard PID.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data for three variables
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    dict
        Dictionary containing PID components across lags, as well as causal metrics
    """
    # Get directed information across multiple lags
    di_results = estimate_multivariate_directed_information(X, Y, Z, max_lag, bins)
    
    # Initialize results structure for PID components
    pid_results = {
        'lags': di_results['lags'],
        'redundancy': np.zeros(max_lag),
        'unique_x': np.zeros(max_lag),
        'unique_y': np.zeros(max_lag),
        'synergy': np.zeros(max_lag),
        'total_di': np.zeros(max_lag)
    }
    
    # Compute standard PID for each lag
    for lag_idx, lag in enumerate(range(1, max_lag + 1)):
        # Use temporal_pid from the imported function to calculate PID components
        pid = temporal_pid(X, Y, Z, lag=lag, bins=bins)
        
        # Store components
        pid_results['redundancy'][lag_idx] = pid['redundancy']
        pid_results['unique_x'][lag_idx] = pid['unique_x1']
        pid_results['unique_y'][lag_idx] = pid['unique_x2']
        pid_results['synergy'][lag_idx] = pid['synergy']
        pid_results['total_di'][lag_idx] = pid['total_di']
    
    # Combine PID and DI results
    results = {**pid_results, **di_results}
    # Add causal structure detection based on lag patterns
    results['structure_scores'] = identify_causal_structure(di_results)
    
    return results

def identify_causal_structure(di_results):
    """
    Identify the most likely causal structure based on directed information patterns.
    
    This function uses a simpler, more direct approach focusing on the key
    distinguishing features of each causal structure.
    
    Parameters:
    -----------
    di_results : dict
        Directed information results from estimate_multivariate_directed_information
        
    Returns:
    --------
    dict
        Scores for each causal structure based on directed information patterns
    """
    # Extract lag information
    max_lag = len(di_results['lags'])
    lags = di_results['lags']
    
    # Initialize structure scores
    structure_scores = {
        'chain_score': np.zeros(max_lag),
        'fork_score': np.zeros(max_lag),
        'v_structure_score': np.zeros(max_lag),
        'inferred_structure': []
    }
    
    for lag_idx, lag in enumerate(lags):
        # Get directed information values
        di_x_to_z = di_results['di_x_to_z'][lag_idx]
        di_z_to_x = di_results['di_z_to_x'][lag_idx]
        di_y_to_z = di_results['di_y_to_z'][lag_idx]
        di_z_to_y = di_results['di_z_to_y'][lag_idx]
        di_x_to_y = di_results['di_x_to_y'][lag_idx]
        di_y_to_x = di_results['di_y_to_x'][lag_idx]
        
        # Simple approach: check directional strengths directly
        
        # Chain (X→Z→Y): Strong X→Z, Z→Y, weak X→Y
        chain_score = di_x_to_z * di_z_to_y / (di_x_to_y + 0.01)
        
        # Fork (X←Z→Y): Strong Z→X, Z→Y, weak X→Y and Y→X
        fork_score = di_z_to_x * di_z_to_y / (di_x_to_y + di_y_to_x + 0.01)
        
        # V-structure (X→Z←Y): Strong X→Z, Y→Z, weak Z→X, Z→Y
        v_structure_score = di_x_to_z * di_y_to_z / (di_z_to_x + di_z_to_y + 0.01)
        
        # Store scores
        structure_scores['chain_score'][lag_idx] = chain_score
        structure_scores['fork_score'][lag_idx] = fork_score
        structure_scores['v_structure_score'][lag_idx] = v_structure_score
        
        # Determine most likely structure for this lag
        scores = {
            'Chain': chain_score,
            'Fork': fork_score,
            'V-structure': v_structure_score
        }
        structure_scores['inferred_structure'].append(max(scores, key=scores.get))
    
    return structure_scores

def multi_lag_causal_pid(X, Y, Z, max_lag=5, bins=10):
    """
    A comprehensive directed information framework for causal structure identification.
    
    This function combines multi-lag partial directed information analysis with
    causal structure detection techniques to provide a detailed view of causal
    relationships between variables.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data for three variables
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    dict
        Comprehensive analysis results including PID components, 
        directed information, and causal structure detection
    """
    # Perform both analyses
    mlpdi_results = multi_lag_partial_directed_information(X, Y, Z, max_lag, bins)
    std_pid_result = temporal_pid(X, Y, Z, lag=1, bins=bins)
    
    # Calculate divergence from standard PID for each lag
    pid_divergence = np.zeros(max_lag)
    for lag_idx in range(max_lag):
        # Calculate Euclidean distance between standard PID and directed PID components
        std_components = np.array([
            std_pid_result['redundancy'],
            std_pid_result['unique_x1'],
            std_pid_result['unique_x2'],
            std_pid_result['synergy']
        ])
        
        lag_components = np.array([
            mlpdi_results['redundancy'][lag_idx],
            mlpdi_results['unique_x'][lag_idx],
            mlpdi_results['unique_y'][lag_idx],
            mlpdi_results['synergy'][lag_idx]
        ])
        
        # Calculate normalized distance
        pid_divergence[lag_idx] = np.linalg.norm(std_components - lag_components) / np.linalg.norm(std_components)
    
    # Add divergence to results
    mlpdi_results['pid_divergence'] = pid_divergence
    
    # Add time-aggregated causal structure inference
    structure_counts = {
        'Chain': mlpdi_results['structure_scores']['inferred_structure'].count('Chain'),
        'Fork': mlpdi_results['structure_scores']['inferred_structure'].count('Fork'),
        'V-structure': mlpdi_results['structure_scores']['inferred_structure'].count('V-structure')
    }
    
    # Determine overall structure based on majority vote
    overall_structure = max(structure_counts, key=structure_counts.get)
    
    # Calculate confidence based on proportion of lags that agree
    confidence = structure_counts[overall_structure] / max_lag
    
    mlpdi_results['overall_structure'] = overall_structure
    mlpdi_results['structure_confidence'] = confidence
    mlpdi_results['structure_counts'] = structure_counts
    
    return mlpdi_results

def compare_pid_frameworks(X, Y, Z, max_lag=5, bins=10):
    """
    Compare standard PID and multi-lag directed information frameworks.
    
    This function analyzes the same data using both standard PID and the
    multi-lag directed information framework to highlight the advantages
    of temporal analysis.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data for three variables
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    dict
        Comparison results between standard and multi-lag PID
    """
    # Calculate standard PID
    std_pid = temporal_pid(X, Y, Z, lag=1, bins=bins)
    
    # Calculate multi-lag directed PID
    mlpdi = multi_lag_causal_pid(X, Y, Z, max_lag, bins)
    
    # Return comparison results
    return {
        'standard_pid': std_pid,
        'multi_lag_pid': mlpdi,
        'advantages': {
            'causal_detection': mlpdi['overall_structure'],
            'confidence': mlpdi['structure_confidence'],
            'max_information_lag': mlpdi['lags'][np.argmax(mlpdi['total_di'])],
            'max_divergence_lag': mlpdi['lags'][np.argmax(mlpdi['pid_divergence'])],
        }
    }

def plot_causal_structure_analysis(X, Y, Z, max_lag=5, bins=10):
    """
    Visualize the results of the multi-lag directed information analysis.
    
    Creates comprehensive visualizations showing how causal relationships
    evolve across different time lags and compares standard PID with
    the multi-lag directed PID framework.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data for three variables
    max_lag : int, default=5
        Maximum time lag to consider
    bins : int, default=10
        Number of bins for discretization
    """
    # Get results from both frameworks
    comparison = compare_pid_frameworks(X, Y, Z, max_lag, bins)
    std_pid = comparison['standard_pid']
    mlpdi = comparison['multi_lag_pid']
    
    # Create figure with subplots
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Standard PID vs. Multi-Lag PID at lag 1
    plt.subplot(3, 2, 1)
    components = ['Redundancy', 'Unique X', 'Unique Y', 'Synergy']
    std_values = [std_pid['redundancy'], std_pid['unique_x1'], std_pid['unique_x2'], std_pid['synergy']]
    ml_values = [mlpdi['redundancy'][0], mlpdi['unique_x'][0], mlpdi['unique_y'][0], mlpdi['synergy'][0]]
    
    x = np.arange(len(components))
    width = 0.35
    
    plt.bar(x - width/2, std_values, width, label='Standard PID')
    plt.bar(x + width/2, ml_values, width, label='Multi-Lag PID (lag=1)')
    
    plt.ylabel('Information (bits)')
    plt.title('Standard PID vs. Multi-Lag PID (lag=1)')
    plt.xticks(x, components)
    plt.legend()
    
    # Plot 2: PID Components across lags
    plt.subplot(3, 2, 2)
    lags = mlpdi['lags']
    plt.plot(lags, mlpdi['redundancy'], 'b.-', label='Redundancy')
    plt.plot(lags, mlpdi['unique_x'], 'g.-', label='Unique X')
    plt.plot(lags, mlpdi['unique_y'], 'r.-', label='Unique Y')
    plt.plot(lags, mlpdi['synergy'], 'm.-', label='Synergy')
    plt.plot(lags, mlpdi['total_di'], 'k-', linewidth=2, label='Total DI')
    
    plt.xlabel('Time Lag')
    plt.ylabel('Information (bits)')
    plt.title('Multi-Lag PID Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Directed Information Matrix Heatmap
    plt.subplot(3, 2, 3)
    di_matrix = np.zeros((6, max_lag))
    di_matrix[0, :] = mlpdi['di_x_to_z']
    di_matrix[1, :] = mlpdi['di_z_to_x']
    di_matrix[2, :] = mlpdi['di_y_to_z']
    di_matrix[3, :] = mlpdi['di_z_to_y']
    di_matrix[4, :] = mlpdi['di_x_to_y']
    di_matrix[5, :] = mlpdi['di_y_to_x']
    
    plt.imshow(di_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Directed Information (bits)')
    plt.yticks(range(6), ['X→Z', 'Z→X', 'Y→Z', 'Z→Y', 'X→Y', 'Y→X'])
    plt.xticks(range(max_lag), [f'Lag {l}' for l in lags])
    plt.title('Directed Information Matrix')
    
    # Plot 4: Causal Structure Scores across lags
    plt.subplot(3, 2, 4)
    plt.plot(lags, mlpdi['structure_scores']['chain_score'], 'b.-', label='Chain')
    plt.plot(lags, mlpdi['structure_scores']['fork_score'], 'g.-', label='Fork')
    plt.plot(lags, mlpdi['structure_scores']['v_structure_score'], 'r.-', label='V-structure')
    
    plt.xlabel('Time Lag')
    plt.ylabel('Structure Score')
    plt.title('Causal Structure Scores by Lag')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Divergence from Standard PID
    plt.subplot(3, 2, 5)
    plt.bar(lags, mlpdi['pid_divergence'], color='purple')
    plt.xlabel('Time Lag')
    plt.ylabel('Normalized Divergence')
    plt.title('Divergence from Standard PID')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Structure Inference Confidence
    plt.subplot(3, 2, 6)
    structures = list(mlpdi['structure_counts'].keys())
    counts = list(mlpdi['structure_counts'].values())
    
    plt.bar(structures, counts, color=['blue', 'green', 'red'])
    plt.axhline(y=max_lag/2, linestyle='--', color='gray', alpha=0.5)
    plt.text(0.5, max_lag/2 + 0.1, 'Majority Threshold', 
             ha='center', va='bottom', color='gray')
    
    plt.xlabel('Causal Structure')
    plt.ylabel('Count (across lags)')
    plt.title(f'Inferred Structure: {mlpdi["overall_structure"]} (Confidence: {mlpdi["structure_confidence"]:.2f})')
    
    plt.tight_layout()
    plt.savefig('results/multi_lag_causal_pid_analysis.png', dpi=300, bbox_inches='tight')
    print("Analysis figure saved to 'results/multi_lag_causal_pid_analysis.png'")
    
    return comparison

#----------------------------------------------------------------------------------
# Testing and Analysis Functions
#----------------------------------------------------------------------------------

def test_multi_lag_pid_framework():
    """
    Test the multi-lag directed information framework on synthetic datasets.
    
    This function generates data from different causal structures and analyzes them
    using both standard PID and multi-lag directed PID approaches to demonstrate
    the advantages of the latter.
    """
    print("Testing Multi-Lag Directed Information Framework...")
    
    # Set up parameters
    n_samples = 1000
    noise_level = 0.1
    max_lag = 10
    bins = 10
    seed = 2023
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("Generating data for different causal structures...")
    
    # Generate data for each causal structure
    X_chain, Z_chain, Y_chain = generate_chain_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    print("Chain structure data generated.")
    
    X_fork, Z_fork, Y_fork = generate_fork_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    print("Fork structure data generated.")
    
    X_v, Z_v, Y_v = generate_v_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    print("V-structure data generated.")
    
    # Analyze each structure
    print("\nAnalyzing Chain Structure (X → Z → Y)...")
    chain_results = plot_causal_structure_analysis(X_chain, Y_chain, Z_chain, max_lag=max_lag, bins=bins)
    
    print("\nAnalyzing Fork Structure (X ← Z → Y)...")
    fork_results = plot_causal_structure_analysis(X_fork, Y_fork, Z_fork, max_lag=max_lag, bins=bins)
    
    print("\nAnalyzing V-Structure (X → Z ← Y)...")
    v_results = plot_causal_structure_analysis(X_v, Y_v, Z_v, max_lag=max_lag, bins=bins)
    
    # Create overall comparison figure
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Structure Inference Accuracy
    plt.subplot(2, 2, 1)
    structures = ['Chain', 'Fork', 'V-structure']
    expected = structures  # Expected ground truth
    inferred_std = ['Unknown', 'Unknown', 'Unknown']  # Standard PID can't determine this
    inferred_ml = [
        chain_results['multi_lag_pid']['overall_structure'],
        fork_results['multi_lag_pid']['overall_structure'],
        v_results['multi_lag_pid']['overall_structure']
    ]
    
    # Calculate accuracy
    accuracy_std = [1 if inferred_std[i] == expected[i] else 0 for i in range(3)]
    accuracy_ml = [1 if inferred_ml[i] == expected[i] else 0 for i in range(3)]
    
    x = np.arange(len(structures))
    width = 0.35
    
    plt.bar(x - width/2, accuracy_std, width, label='Standard PID')
    plt.bar(x + width/2, accuracy_ml, width, label='Multi-Lag DI-PID')
    
    plt.ylabel('Correct Inference (1=Yes, 0=No)')
    plt.title('Structure Inference Accuracy')
    plt.xticks(x, structures)
    plt.ylim(0, 1.2)
    plt.legend()
    
    # Plot 2: Inference Confidence
    plt.subplot(2, 2, 2)
    confidence = [
        chain_results['multi_lag_pid']['structure_confidence'],
        fork_results['multi_lag_pid']['structure_confidence'],
        v_results['multi_lag_pid']['structure_confidence']
    ]
    
    plt.bar(structures, confidence, color='green')
    plt.axhline(y=0.6, linestyle='--', color='red', alpha=0.5)
    plt.text(0.5, 0.62, 'High Confidence Threshold', ha='center', color='red')
    
    plt.ylabel('Confidence')
    plt.title('Inference Confidence by Structure')
    plt.ylim(0, 1.1)
    
    # Plot 3: PID Divergence by Structure
    plt.subplot(2, 2, 3)
    
    # Maximum divergence for each structure
    max_div_chain = max(chain_results['multi_lag_pid']['pid_divergence'])
    max_div_fork = max(fork_results['multi_lag_pid']['pid_divergence'])
    max_div_v = max(v_results['multi_lag_pid']['pid_divergence'])
    
    plt.bar(structures, [max_div_chain, max_div_fork, max_div_v], color='purple')
    
    plt.ylabel('Maximum Divergence from Standard PID')
    plt.title('Information Revealed by Multi-Lag Analysis')
    
    # Plot 4: Optimal Lag by Structure
    plt.subplot(2, 2, 4)
    
    # Lag with maximum total directed information
    optimal_lag_chain = chain_results['advantages']['max_information_lag']
    optimal_lag_fork = fork_results['advantages']['max_information_lag']
    optimal_lag_v = v_results['advantages']['max_information_lag']
    
    plt.bar(structures, [optimal_lag_chain, optimal_lag_fork, optimal_lag_v], color='orange')
    
    plt.ylabel('Optimal Time Lag')
    plt.title('Time Lag with Maximum Information')
    
    plt.tight_layout()
    plt.savefig('results/multi_lag_framework_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison figure saved to 'results/multi_lag_framework_comparison.png'")
    
    # Return comprehensive results
    return {
        'chain': chain_results,
        'fork': fork_results,
        'v_structure': v_results,
        'accuracy': sum(accuracy_ml) / len(accuracy_ml),
        'average_confidence': sum(confidence) / len(confidence)
    }

#----------------------------------------------------------------------------------
# Enhanced Causal PID Framework
#----------------------------------------------------------------------------------

def directional_pid(X, Y, Z, lag=1, bins=10):
    """
    Enhanced PID framework that considers directionality to better distinguish 
    causal structures.
    
    This function computes both standard PID and additional causal metrics to
    create a signature for different causal structures.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data for the three variables
    lag : int, default=1
        Time lag to consider for causal influence
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    dict
        Dictionary containing standard PID components plus causal metrics
    """
    # Compute standard PID
    pid_result = temporal_pid(X, Y, Z, lag=lag, bins=bins)
    
    # Compute transfer entropies (causal information flow)
    te_x_to_z = estimate_transfer_entropy(X, Z, lag=lag, bins=bins)
    te_z_to_x = estimate_transfer_entropy(Z, X, lag=lag, bins=bins)
    te_y_to_z = estimate_transfer_entropy(Y, Z, lag=lag, bins=bins)
    te_z_to_y = estimate_transfer_entropy(Z, Y, lag=lag, bins=bins)
    te_x_to_y = estimate_transfer_entropy(X, Y, lag=lag, bins=bins)
    te_y_to_x = estimate_transfer_entropy(Y, X, lag=lag, bins=bins)
    
    # Compute conditional mutual information
    # I(X;Z|Y) - X and Z share information not in Y
    cmi_x_z_given_y = compute_conditional_mi(X, Z, Y, lag=lag, bins=bins)
    # I(Y;Z|X) - Y and Z share information not in X
    cmi_y_z_given_x = compute_conditional_mi(Y, Z, X, lag=lag, bins=bins)
    # I(X;Y|Z) - X and Y share information not in Z
    cmi_x_y_given_z = compute_conditional_mi(X, Y, Z, lag=lag, bins=bins)
    
    # Compute interaction information
    interaction_info = compute_interaction_information(X, Y, Z, lag=lag, bins=bins)
    
    # Asymmetry measures (helpful for distinguishing directionality)
    te_asymmetry_xz = te_x_to_z - te_z_to_x
    te_asymmetry_yz = te_y_to_z - te_z_to_y
    te_asymmetry_xy = te_x_to_y - te_y_to_x
    # pdb.set_trace()
    # Compute causal signatures for each causal structure
    chain_signature = compute_chain_signature(te_x_to_z, te_z_to_y, te_x_to_y, 
                                            cmi_x_z_given_y, cmi_y_z_given_x, cmi_x_y_given_z)
    fork_signature = compute_fork_signature(te_z_to_x, te_z_to_y, te_x_to_y,
                                          cmi_x_z_given_y, cmi_y_z_given_x, cmi_x_y_given_z)
    v_signature = compute_v_signature(te_x_to_z, te_y_to_z, te_x_to_y,
                                     cmi_x_z_given_y, cmi_y_z_given_x, cmi_x_y_given_z)
    
    # Return combined results
    return {
        # Standard PID components
        'redundancy': pid_result['redundancy'],
        'unique_x': pid_result['unique_x1'],
        'unique_y': pid_result['unique_x2'],
        'synergy': pid_result['synergy'],
        
        # Transfer entropy measures
        'te_x_to_z': te_x_to_z,
        'te_z_to_x': te_z_to_x,
        'te_y_to_z': te_y_to_z,
        'te_z_to_y': te_z_to_y,
        'te_x_to_y': te_x_to_y,
        'te_y_to_x': te_y_to_x,
        
        # Transfer entropy asymmetries
        'te_asymmetry_xz': te_asymmetry_xz,
        'te_asymmetry_yz': te_asymmetry_yz,
        'te_asymmetry_xy': te_asymmetry_xy,
        
        # Conditional mutual information
        'cmi_x_z_given_y': cmi_x_z_given_y,
        'cmi_y_z_given_x': cmi_y_z_given_x,
        'cmi_x_y_given_z': cmi_x_y_given_z,
        
        # Interaction information
        'interaction_info': interaction_info,
        
        # Structure signatures (higher = more likely)
        'chain_signature': chain_signature,
        'fork_signature': fork_signature,
        'v_signature': v_signature,
        
        # Inferred causal structure based on highest signature
        'inferred_structure': infer_structure(chain_signature, fork_signature, v_signature)
    }

def compute_conditional_mi(X, Y, Z, lag=1, bins=10):
    """
    Compute conditional mutual information I(X;Y|Z).
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data
    lag : int, default=1
        Time lag
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    float
        Conditional mutual information I(X;Y|Z)
    """
    # Adjust for lag
    X_past = X[:-lag]
    Y_past = Y[:-lag]
    Z_past = Z[:-lag]
    
    # Discretize data
    x_bins = np.linspace(min(X_past), max(X_past), bins+1)
    y_bins = np.linspace(min(Y_past), max(Y_past), bins+1)
    z_bins = np.linspace(min(Z_past), max(Z_past), bins+1)
    
    x_disc = np.digitize(X_past, x_bins)
    y_disc = np.digitize(Y_past, y_bins)
    z_disc = np.digitize(Z_past, z_bins)
    
    # Compute I(X;Y,Z) - I(X;Z)
    # Joint variable XZ
    xz_joint = x_disc * bins + z_disc
    xy_joint = x_disc * bins + y_disc
    xyz_joint = x_disc * bins**2 + y_disc * bins + z_disc
    
    # I(X;Y,Z)
    mi_xyz = mutual_info_score(xyz_joint[:len(x_disc)-bins], x_disc[bins:]) / np.log(2)
    
    # I(X;Z)
    mi_xz = mutual_info_score(xz_joint[:len(x_disc)-bins], x_disc[bins:]) / np.log(2)
    
    # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
    return max(0, mi_xyz - mi_xz)

def compute_interaction_information(X, Y, Z, lag=1, bins=10):
    """
    Compute interaction information I(X;Y;Z).
    
    Interaction information can be positive (synergistic) or negative (redundant).
    Different causal structures yield different interaction information patterns.
    
    Parameters:
    -----------
    X, Y, Z : numpy.ndarray
        Time series data
    lag : int, default=1
        Time lag
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    float
        Interaction information I(X;Y;Z)
    """
    # Adjust for lag
    X_past = X[:-lag]
    Y_past = Y[:-lag]
    Z_present = Z[lag:]
    
    # Discretize data
    x_bins = np.linspace(min(X_past), max(X_past), bins+1)
    y_bins = np.linspace(min(Y_past), max(Y_past), bins+1)
    z_bins = np.linspace(min(Z_present), max(Z_present), bins+1)
    
    x_disc = np.digitize(X_past, x_bins)
    y_disc = np.digitize(Y_past, y_bins)
    z_disc = np.digitize(Z_present, z_bins)
    
    # Compute I(X;Z) - mutual information between X and Z
    xz_joint = x_disc * bins + z_disc
    mi_xz = mutual_info_score(x_disc, z_disc) / np.log(2)
    
    # Compute I(Y;Z) - mutual information between Y and Z
    yz_joint = y_disc * bins + z_disc
    mi_yz = mutual_info_score(y_disc, z_disc) / np.log(2)
    
    # Compute I(X;Y) - mutual information between X and Y
    xy_joint = x_disc * bins + y_disc
    mi_xy = mutual_info_score(x_disc, y_disc) / np.log(2)
    
    # Compute I(X;Y;Z) = I(X;Z) + I(Y;Z) - I(X,Y;Z)
    # Joint variable XY
    xyz_joint = x_disc * bins**2 + y_disc * bins + z_disc
    
    # I(X,Y;Z)
    xy_combined = xy_joint
    mi_xyz = mutual_info_score(xy_combined, z_disc) / np.log(2)
    
    # I(X;Y;Z) = I(X;Z) + I(Y;Z) - I(X,Y;Z)
    return mi_xz + mi_yz - mi_xyz

def compute_chain_signature(te_x_to_z, te_z_to_y, te_x_to_y, cmi_x_z_given_y, cmi_y_z_given_x, cmi_x_y_given_z):
    """
    Compute a signature score for chain structure X→Z→Y.
    
    Chain characteristics:
    - High X→Z and Z→Y transfer entropy
    - Low X→Y transfer entropy (or it should be explained away by Z)
    - Low I(X;Y|Z) (X and Y are conditionally independent given Z)
    - High I(X;Z|Y) and I(Y;Z|X)
    
    Returns:
    --------
    float
        Higher value indicates more likelihood of chain structure
    """
    # Chain signature components
    high_x_to_z = te_x_to_z
    high_z_to_y = te_z_to_y
    low_x_to_y = 1 / (1 + te_x_to_y)  # Inverse relationship
    low_cmi_x_y_given_z = 1 / (1 + cmi_x_y_given_z)  # Inverse relationship
    high_cmi_x_z_given_y = cmi_x_z_given_y
    high_cmi_y_z_given_x = cmi_y_z_given_x
    
    # Combined signature (geometric mean to balance factors)
    signature = (high_x_to_z * high_z_to_y * low_x_to_y * 
                low_cmi_x_y_given_z * high_cmi_x_z_given_y * high_cmi_y_z_given_x) ** (1/6)
    
    return signature

def compute_fork_signature(te_z_to_x, te_z_to_y, te_x_to_y, cmi_x_z_given_y, cmi_y_z_given_x, cmi_x_y_given_z):
    """
    Compute a signature score for fork/common cause structure X←Z→Y.
    
    Fork characteristics:
    - High Z→X and Z→Y transfer entropy
    - Low X→Y and Y→X transfer entropy
    - Low I(X;Y|Z) (X and Y are conditionally independent given Z)
    - High I(X;Z|Y) and I(Y;Z|X)
    
    Returns:
    --------
    float
        Higher value indicates more likelihood of fork structure
    """
    # Fork signature components
    high_z_to_x = te_z_to_x
    high_z_to_y = te_z_to_y
    low_x_to_y = 1 / (1 + te_x_to_y)  # Inverse relationship
    low_cmi_x_y_given_z = 1 / (1 + cmi_x_y_given_z)  # Inverse relationship
    high_cmi_x_z_given_y = cmi_x_z_given_y
    high_cmi_y_z_given_x = cmi_y_z_given_x
    
    # Combined signature (geometric mean to balance factors)
    signature = (high_z_to_x * high_z_to_y * low_x_to_y * 
                low_cmi_x_y_given_z * high_cmi_x_z_given_y * high_cmi_y_z_given_x) ** (1/6)
    
    return signature

def compute_v_signature(te_x_to_z, te_y_to_z, te_x_to_y, cmi_x_z_given_y, cmi_y_z_given_x, cmi_x_y_given_z):
    """
    Compute a signature score for v-structure/collider X→Z←Y.
    
    V-structure characteristics:
    - High X→Z and Y→Z transfer entropy
    - Low or negative X→Y transfer entropy
    - High I(X;Y|Z) (X and Y become dependent when conditioning on Z)
    - High I(X;Z|Y) and I(Y;Z|X)
    
    Returns:
    --------
    float
        Higher value indicates more likelihood of v-structure
    """
    # V-structure signature components
    high_x_to_z = te_x_to_z
    high_y_to_z = te_y_to_z
    low_x_to_y = 1 / (1 + te_x_to_y)  # Inverse relationship
    high_cmi_x_y_given_z = cmi_x_y_given_z  # Explaining away effect
    high_cmi_x_z_given_y = cmi_x_z_given_y
    high_cmi_y_z_given_x = cmi_y_z_given_x
    
    # Combined signature (geometric mean to balance factors)
    signature = (high_x_to_z * high_y_to_z * low_x_to_y * 
                high_cmi_x_y_given_z * high_cmi_x_z_given_y * high_cmi_y_z_given_x) ** (1/6)
    
    return signature

def infer_structure(chain_sig, fork_sig, v_sig):
    """
    Infer the most likely causal structure based on signature scores.
    
    Returns:
    --------
    str
        'Chain', 'Fork', or 'V-structure'
    """
    scores = {
        'Chain': chain_sig,
        'Fork': fork_sig,
        'V-structure': v_sig
    }
    
    return max(scores, key=scores.get)

#----------------------------------------------------------------------------------
# Data Generation Functions (Same as in simple_causal_test.py)
#----------------------------------------------------------------------------------

def generate_chain_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a chain structure: X → Z → Y with temporal ordering.
    
    This implementation creates data with clear temporal patterns:
    - X influences Z with a lag
    - Z influences Y with a lag
    - X has a weak direct effect on Y (mediated through Z)
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'x_to_z': 0.8, 'z_to_y': 0.8, 'x_to_y': 0.2}
    
    # Initialize arrays with random noise
    X = np.random.randn(n_samples)
    Z = np.zeros(n_samples)
    Y = np.zeros(n_samples)
    
    # Add temporal dependencies
    for t in range(1, n_samples):
        # Z depends on previous X
        Z[t] = coefficients['x_to_z'] * X[t-1] + np.random.randn() * noise_level
        
        # Y depends on previous Z, and slightly on previous X (indirect path)
        if t > 1:
            Y[t] = coefficients['z_to_y'] * Z[t-1] + coefficients['x_to_y'] * X[t-2] + np.random.randn() * noise_level
    
    return X, Z, Y

def generate_fork_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a fork/common cause structure: X ← Z → Y with temporal ordering.
    
    This implementation creates data with clear temporal patterns:
    - Z influences both X and Y with a lag
    - X and Y have no direct causal relationship
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'z_to_x': 0.8, 'z_to_y': 0.8}
    
    # Initialize arrays
    Z = np.random.randn(n_samples)
    X = np.zeros(n_samples)
    Y = np.zeros(n_samples)
    
    # Add temporal dependencies
    for t in range(1, n_samples):
        # X depends on previous Z
        X[t] = coefficients['z_to_x'] * Z[t-1] + np.random.randn() * noise_level
        
        # Y depends on previous Z
        Y[t] = coefficients['z_to_y'] * Z[t-1] + np.random.randn() * noise_level
    
    return X, Z, Y

def generate_v_structure(n_samples=1000, coefficients=None, noise_level=0.1, seed=None):
    """
    Generate data from a v-structure/collider: X → Z ← Y with temporal ordering.
    
    This implementation creates data with clear temporal patterns:
    - Both X and Y influence Z with a lag
    - X and Y have no direct causal relationship
    """
    if seed is not None:
        np.random.seed(seed)
        
    if coefficients is None:
        coefficients = {'x_to_z': 0.8, 'y_to_z': 0.8}
    
    # Initialize arrays - X and Y are independent sources
    X = np.random.randn(n_samples)
    Y = np.random.randn(n_samples)
    Z = np.zeros(n_samples)
    
    # Add temporal dependencies
    for t in range(1, n_samples):
        # Z depends on previous X and previous Y (collider)
        Z[t] = coefficients['x_to_z'] * X[t-1] + coefficients['y_to_z'] * Y[t-1] + np.random.randn() * noise_level
    
    return X, Z, Y

#----------------------------------------------------------------------------------
# Testing the Enhanced Framework
#----------------------------------------------------------------------------------

def test_and_compare_causal_structures():
    """
    Test the enhanced causal PID framework on different causal structures.
    """
    print("Testing Enhanced Causal PID Framework...")
    
    # Set up parameters
    n_samples = 1000
    noise_level = 0.1
    lag = 4
    bins = 10
    seed = 5
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("Generating data for different causal structures...")
    
    # Generate data for each causal structure
    X_chain, Z_chain, Y_chain = generate_chain_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    print("Chain structure data generated.")
    
    X_fork, Z_fork, Y_fork = generate_fork_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    print("Fork structure data generated.")
    
    X_v, Z_v, Y_v = generate_v_structure(
        n_samples=n_samples, noise_level=noise_level, seed=seed)
    print("V-structure data generated.")
    
    # Test standard PID for comparison
    print("\nAnalyzing with standard PID...")
    pid_chain = temporal_pid(X_chain, Y_chain, Z_chain, lag=lag, bins=bins)
    pid_fork = temporal_pid(X_fork, Y_fork, Z_fork, lag=lag, bins=bins)
    pid_v = temporal_pid(X_v, Y_v, Z_v, lag=lag, bins=bins)
    
    # Store standard PID results
    std_pid_results = {
        'Chain': pid_chain,
        'Fork': pid_fork,
        'V-structure': pid_v
    }
    
    # Test enhanced causal PID
    print("\nAnalyzing with Enhanced Causal PID...")
    start_time = time.time()
    
    print("Analyzing Chain Structure (X → Z → Y)...")
    causal_pid_chain = directional_pid(X_chain, Y_chain, Z_chain, lag=lag, bins=bins)
    print(f"Inferred structure: {causal_pid_chain['inferred_structure']}")
    
    print("\nAnalyzing Fork Structure (X ← Z → Y)...")
    causal_pid_fork = directional_pid(X_fork, Y_fork, Z_fork, lag=lag, bins=bins)
    print(f"Inferred structure: {causal_pid_fork['inferred_structure']}")
    
    print("\nAnalyzing V-Structure (X → Z ← Y)...")
    causal_pid_v = directional_pid(X_v, Y_v, Z_v, lag=lag, bins=bins)
    print(f"Inferred structure: {causal_pid_v['inferred_structure']}")
    
    print(f"\nEnhanced causal PID analysis took {time.time() - start_time:.2f} seconds")
    
    # Store enhanced PID results
    causal_pid_results = {
        'Chain': causal_pid_chain,
        'Fork': causal_pid_fork,
        'V-structure': causal_pid_v
    }
    
    # Create visualizations
    plot_comparison(std_pid_results, causal_pid_results)
    
    return std_pid_results, causal_pid_results

def plot_comparison(std_pid_results, causal_pid_results):
    """
    Create visualizations comparing standard PID and enhanced causal PID.
    """
    # Plot standard PID components
    plt.figure(figsize=(14, 9))
    
    # Plot 1: Standard PID components by structure
    plt.subplot(2, 2, 1)
    
    # Standard PID components
    components = ['redundancy', 'unique_x1', 'unique_x2', 'synergy']
    structures = list(std_pid_results.keys())
    
    # Set up bar positions
    x = np.arange(len(components))
    width = 0.25
    
    # Plot bars for each structure
    for i, structure in enumerate(structures):
        values = [std_pid_results[structure][c] for c in components]
        plt.bar(x + i*width, values, width, label=structure)
    
    plt.ylabel('Information (bits)')
    plt.title('Standard PID Components')
    plt.xticks(x + width, ['Redundancy', 'Unique X', 'Unique Y', 'Synergy'])
    plt.legend()
    
    # Plot 2: Transfer entropy asymmetry
    plt.subplot(2, 2, 2)
    
    # Asymmetry metrics
    asymmetry_metrics = ['te_asymmetry_xz', 'te_asymmetry_yz', 'te_asymmetry_xy']
    x = np.arange(len(asymmetry_metrics))
    
    # Plot bars for each structure
    for i, structure in enumerate(structures):
        values = [causal_pid_results[structure][m] for m in asymmetry_metrics]
        plt.bar(x + i*width, values, width, label=structure)
    
    plt.ylabel('Transfer Entropy Asymmetry')
    plt.title('Directional Information Flow')
    plt.xticks(x + width, ['X-Z', 'Y-Z', 'X-Y'])
    plt.legend()
    
    # Plot 3: Conditional mutual information
    plt.subplot(2, 2, 3)
    
    # CMI metrics
    cmi_metrics = ['cmi_x_z_given_y', 'cmi_y_z_given_x', 'cmi_x_y_given_z']
    x = np.arange(len(cmi_metrics))
    
    # Plot bars for each structure
    for i, structure in enumerate(structures):
        values = [causal_pid_results[structure][m] for m in cmi_metrics]
        plt.bar(x + i*width, values, width, label=structure)
    
    plt.ylabel('Conditional Mutual Information')
    plt.title('Conditional Independence Patterns')
    plt.xticks(x + width, ['I(X;Z|Y)', 'I(Y;Z|X)', 'I(X;Y|Z)'])
    plt.legend()
    
    # Plot 4: Structure signatures
    plt.subplot(2, 2, 4)
    
    # Structure signatures
    signature_metrics = ['chain_signature', 'fork_signature', 'v_signature']
    x = np.arange(len(signature_metrics))
    
    # Plot bars for each structure
    for i, structure in enumerate(structures):
        values = [causal_pid_results[structure][m] for m in signature_metrics]
        plt.bar(x + i*width, values, width, label=structure)
    
    plt.ylabel('Signature Score')
    plt.title('Causal Structure Signatures')
    plt.xticks(x + width, ['Chain', 'Fork', 'V-structure'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/causal_pid_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison figure saved to 'results/causal_pid_comparison.png'")
    
    # Summary table
    plt.figure(figsize=(12, 6))
    structures = list(causal_pid_results.keys())
    
    # Format inference accuracy data
    accuracy_data = []
    for struct in structures:
        # Check if true structure was inferred correctly
        inferred = causal_pid_results[struct]['inferred_structure']
        accuracy_data.append(1 if inferred == struct else 0)
    
    plt.bar(structures, accuracy_data, color=['green' if a == 1 else 'red' for a in accuracy_data])
    plt.ylabel('Correct Inference (1=Yes, 0=No)')
    plt.title('Structure Inference Accuracy')
    plt.ylim(0, 1.2)
    
    # Add text annotations
    for i, v in enumerate(accuracy_data):
        inferred = causal_pid_results[structures[i]]['inferred_structure']
        plt.text(i, v + 0.1, f"Inferred: {inferred}", ha='center')
    
    plt.tight_layout()
    plt.savefig('results/causal_pid_accuracy.png', dpi=300)
    print("Accuracy figure saved to 'results/causal_pid_accuracy.png'")


if __name__ == "__main__":
    # Test the multi-lag directed information framework
    # print("\n=== Testing Multi-Lag Directed Information Framework ===")
    # ml_results = test_multi_lag_pid_framework()
    # print(f"Structure detection accuracy: {ml_results['accuracy'] * 100:.1f}%")
    # print(f"Average detection confidence: {ml_results['average_confidence'] * 100:.1f}%")
    
    # Also test the enhanced causal PID framework for comparison
    print("\n=== Testing Enhanced Causal PID Framework ===")
    std_pid_results, causal_pid_results = test_and_compare_causal_structures() 