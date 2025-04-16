#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directed Information-based Causal Discovery with Time-Lagged Relationships

This script implements methods to discover causal structures using directed information,
accounting for different time lags across variable pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from itertools import product
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_time_series(causal_structure, time_lags, noise_level=0.1, length=1000):
    """
    Generate multivariate time series data with specific causal structure and time lags.
    
    Parameters:
    -----------
    causal_structure : dict
        Dictionary where keys are (from_var, to_var) tuples and values are causal strengths
    time_lags : dict
        Dictionary where keys are (from_var, to_var) tuples and values are time lags
    noise_level : float
        Standard deviation of the noise
    length : int
        Length of the time series
    
    Returns:
    --------
    data : numpy.ndarray
        Generated time series data
    """
    # Identify all variables
    all_vars = set()
    for from_var, to_var in causal_structure.keys():
        all_vars.add(from_var)
        all_vars.add(to_var)
    
    n_vars = len(all_vars)
    var_map = {var: idx for idx, var in enumerate(sorted(all_vars))}
    
    # Get maximum time lag
    max_lag = max(time_lags.values()) if time_lags else 1
    
    # Initialize data with noise
    data = np.random.normal(0, noise_level, (length, n_vars))
    
    # Fill in data based on causal relationships
    for t in range(max_lag, length):
        for (from_var, to_var), strength in causal_structure.items():
            lag = time_lags.get((from_var, to_var), 1)
            from_idx = var_map[from_var]
            to_idx = var_map[to_var]
            
            # Add the causal effect
            data[t, to_idx] += strength * data[t-lag, from_idx]
    
    return data, var_map

def entropy(x):
    """
    Simple histogram-based entropy estimator.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input data
    
    Returns:
    --------
    entropy : float
        Estimated entropy
    """
    # Reshape data
    x = x.reshape(-1, 1) if len(x.shape) == 1 else x
    
    # Bin the data (adaptive bins based on data range)
    n_samples = len(x)
    n_bins = min(int(np.sqrt(n_samples)), 50)  # Rule of thumb for binning
    
    # Calculate entropy
    hist, _ = np.histogramdd(x, bins=n_bins)
    hist = hist / n_samples  # Normalize to get probabilities
    hist = hist[hist > 0]  # Only consider non-zero probabilities
    
    return -np.sum(hist * np.log2(hist))

def conditional_entropy(y, x):
    """
    Calculate conditional entropy H(Y|X) using histogram-based approach.
    
    Parameters:
    -----------
    y : numpy.ndarray
        The dependent variable
    x : numpy.ndarray
        The conditioning variable(s)
    
    Returns:
    --------
    cond_entropy : float
        Conditional entropy value
    """
    # Ensure proper shapes
    y = y.reshape(-1, 1) if len(y.shape) == 1 else y
    x = x.reshape(-1, 1) if len(x.shape) == 1 and x.ndim == 1 else x
    
    # Joint variable
    xy = np.hstack([x, y])
    
    # Entropy of joint distribution minus entropy of conditioning variable
    hxy = entropy(xy)
    hx = entropy(x)
    
    # H(Y|X) = H(X,Y) - H(X)
    return hxy - hx

def transfer_entropy(x, y, k=1, l=1):
    """
    Calculate transfer entropy from X to Y.
    Transfer entropy measures the directional information flow from X to Y.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Source time series
    y : numpy.ndarray
        Target time series
    k : int
        History length for target variable
    l : int
        History length for source variable
    
    Returns:
    --------
    te : float
        Transfer entropy value
    """
    # Ensure x and y are 1D arrays
    x = x.flatten()
    y = y.flatten()
    
    # We need at least k+1 points for y and l points for x
    min_len = max(k+1, l)
    if len(x) <= min_len or len(y) <= min_len:
        return 0
    
    # Create history of target variable (y)
    y_current = y[k:]  # Current value: y_t
    y_history = np.zeros((len(y) - k, k))
    for i in range(k):
        y_history[:, i] = y[k-i-1:-i-1]  # y_t-1, y_t-2, ..., y_t-k
    
    # Create history of source variable (x)
    x_history = np.zeros((len(x) - k, l))
    for i in range(l):
        if i+k < len(x):
            x_history[:, i] = x[k-i-1:-i-1]  # x_t-1, x_t-2, ..., x_t-l
    
    # Ensure all arrays have the same length
    min_length = min(len(y_current), len(y_history), len(x_history))
    y_current = y_current[:min_length]
    y_history = y_history[:min_length]
    x_history = x_history[:min_length]
    
    # Calculate the conditional entropy components
    h_y_given_y_history = conditional_entropy(y_current, y_history)
    h_y_given_xy_history = conditional_entropy(y_current, np.hstack([y_history, x_history]))
    
    # Transfer entropy is the difference of these conditional entropies
    # TE(X->Y) = H(Y|Y_history) - H(Y|Y_history,X_history)
    te = h_y_given_y_history - h_y_given_xy_history
    
    return max(0, te)  # Ensure non-negative

def directed_information(x, y, max_lag=5):
    """
    Calculate directed information from X to Y using transfer entropy.
    Directed information is estimated by finding the optimal lag that 
    maximizes the transfer entropy.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Source time series
    y : numpy.ndarray
        Target time series
    max_lag : int
        Maximum lag to consider
    
    Returns:
    --------
    di : float
        Directed information value (maximum transfer entropy)
    optimal_lag : int
        Lag that maximizes transfer entropy
    """
    n = len(x)
    te_values = []
    
    for lag in range(1, min(max_lag + 1, n // 4)):
        # Shift x by lag to align with y
        x_lagged = x[:-lag] if lag > 0 else x
        y_current = y[lag:] if lag > 0 else y
        
        # Ensure both series have the same length
        min_len = min(len(x_lagged), len(y_current))
        if min_len <= 5:  # Need a minimum number of points for reliable estimation
            continue
            
        x_lagged = x_lagged[-min_len:]
        y_current = y_current[-min_len:]
        
        # Calculate transfer entropy with specific lag
        # Use k=1, l=1 for simplicity (can be extended to use more history)
        te = transfer_entropy(x_lagged, y_current, k=1, l=1)
        te_values.append(te)
    
    if not te_values:
        return 0, 0
    
    # Find the lag that maximizes transfer entropy
    optimal_lag = np.argmax(te_values) + 1
    return te_values[optimal_lag - 1], optimal_lag

def discover_causal_structure(data, var_names=None, max_lag=5, threshold=0.08):
    """
    Discover causal structure using transfer entropy-based directed information.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Multivariate time series data
    var_names : list, optional
        Names of variables
    max_lag : int
        Maximum lag to consider
    threshold : float
        Threshold for considering a causal relationship significant
    
    Returns:
    --------
    causal_graph : networkx.DiGraph
        Discovered causal graph
    discovered_lags : dict
        Dictionary of discovered optimal lags
    di_values : dict
        Dictionary of directed information values
    """
    n_vars = data.shape[1]
    
    if var_names is None:
        var_names = [f'X{i}' for i in range(n_vars)]
    
    # Initialize results
    causal_graph = nx.DiGraph()
    for var in var_names:
        causal_graph.add_node(var)
    
    discovered_lags = {}
    di_values = {}
    
    # Check all possible pairs
    print("Computing transfer entropy-based directed information for all pairs...")
    # First compute all directed information values
    for i, j in product(range(n_vars), range(n_vars)):
        if i != j:  # Don't check self-causality
            print(f"Processing {var_names[i]} → {var_names[j]}", end='\r')
            di_ij, lag_ij = directed_information(data[:, i], data[:, j], max_lag)
            
            di_values[(var_names[i], var_names[j])] = di_ij
    
    # Then add edges, handling bidirectional relationships
    for i, j in product(range(n_vars), range(n_vars)):
        if i != j:  # Don't check self-causality
            di_ij = di_values.get((var_names[i], var_names[j]), 0)
            di_ji = di_values.get((var_names[j], var_names[i]), 0)
            
            # Check if both directions have significant directed information
            if di_ij > threshold and di_ji > threshold:
                # Only keep the direction with greater directed information
                if di_ij > di_ji:
                    causal_graph.add_edge(var_names[i], var_names[j], weight=di_ij)
                    discovered_lags[(var_names[i], var_names[j])] = directed_information(data[:, i], data[:, j], max_lag)[1]
                else:
                    causal_graph.add_edge(var_names[j], var_names[i], weight=di_ji)
                    discovered_lags[(var_names[j], var_names[i])] = directed_information(data[:, j], data[:, i], max_lag)[1]
            # If only one direction is significant, add that edge
            elif di_ij > threshold:
                causal_graph.add_edge(var_names[i], var_names[j], weight=di_ij)
                discovered_lags[(var_names[i], var_names[j])] = directed_information(data[:, i], data[:, j], max_lag)[1]
            elif di_ji > threshold:
                causal_graph.add_edge(var_names[j], var_names[i], weight=di_ji)
                discovered_lags[(var_names[j], var_names[i])] = directed_information(data[:, j], data[:, i], max_lag)[1]
    
    print("\nDone computing transfer entropy-based directed information.")
    return causal_graph, discovered_lags, di_values

def plot_causal_graph(graph, discovered_lags=None):
    """
    Plot the causal graph.
    
    Parameters:
    -----------
    graph : networkx.DiGraph
        Causal graph to plot
    discovered_lags : dict, optional
        Dictionary of discovered optimal lags
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    edges = graph.edges(data=True)
    edge_weights = [d['weight'] * 3 for _, _, d in edges]
    nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.7, 
                          arrowstyle='->', arrowsize=20, edge_color='blue')
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    # Add lag information if available
    if discovered_lags:
        edge_labels = {(u, v): f"lag={discovered_lags.get((u, v), '?')}" 
                      for u, v in graph.edges()}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title('Discovered Causal Graph', fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def evaluate_performance(true_structure, discovered_structure, time_lags, discovered_lags, var_map=None):
    """
    Evaluate the performance of the causal discovery algorithm.
    
    Parameters:
    -----------
    true_structure : dict
        Dictionary of true causal relationships (from_var, to_var) -> strength
    discovered_structure : networkx.DiGraph
        Discovered causal graph
    time_lags : dict
        Dictionary of true time lags
    discovered_lags : dict
        Dictionary of discovered time lags
    var_map : dict, optional
        Mapping from variable names to indices
    
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    true_pairs = set(true_structure.keys())
    disc_pairs = set(discovered_structure.edges())
    
    # True positives, false positives, false negatives
    tp = len(true_pairs.intersection(disc_pairs))
    fp = len(disc_pairs - true_pairs)
    fn = len(true_pairs - disc_pairs)
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Evaluate lag accuracy
    correct_lags = 0
    total_matches = 0
    
    for edge in true_pairs.intersection(disc_pairs):
        total_matches += 1
        if discovered_lags.get(edge) == time_lags.get(edge):
            correct_lags += 1
    
    lag_accuracy = correct_lags / total_matches if total_matches > 0 else 0
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Lag Accuracy': lag_accuracy
    }
    
    return metrics

def test_on_multiple_structures():
    """
    Test the causal discovery method on multiple causal structures.
    """
    # Test cases with different causal structures
    test_cases = [
        {
            'name': 'Simple Chain X → Y → Z',
            'structure': {('X', 'Y'): 0.8, ('Y', 'Z'): 0.7},
            'time_lags': {('X', 'Y'): 2, ('Y', 'Z'): 1}
        },
        {
            'name': 'Common Cause X → Y, X → Z',
            'structure': {('X', 'Y'): 0.8, ('X', 'Z'): 0.7},
            'time_lags': {('X', 'Y'): 1, ('X', 'Z'): 3}
        },
        {
            'name': 'Cycle X → Y → Z → X',
            'structure': {('X', 'Y'): 0.6, ('Y', 'Z'): 0.7, ('Z', 'X'): 0.5},
            'time_lags': {('X', 'Y'): 2, ('Y', 'Z'): 1, ('Z', 'X'): 3}
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nRunning test case: {test_case['name']}")
        
        # Generate data (shorter length for quicker processing)
        data, var_map = generate_time_series(
            test_case['structure'], 
            test_case['time_lags'], 
            noise_level=0.1, 
            length=1000  # Reduced length for faster computation
        )
        
        # Discover causal structure
        var_names = sorted(var_map.keys())
        causal_graph, discovered_lags, di_values = discover_causal_structure(
            data, var_names, max_lag=5, threshold=0.02
        )
        
        # Evaluate performance
        metrics = evaluate_performance(
            test_case['structure'], 
            causal_graph, 
            test_case['time_lags'], 
            discovered_lags,
            var_map
        )
        
        # Save results
        results.append({
            'Test Case': test_case['name'],
            **metrics
        })
        
        # Plot the ground truth and discovered graphs
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        true_graph = nx.DiGraph()
        for var in var_map.keys():
            true_graph.add_node(var)
        for (i, j), strength in test_case['structure'].items():
            true_graph.add_edge(i, j, weight=strength)
        
        pos = nx.spring_layout(true_graph, seed=42)
        nx.draw(true_graph, pos, with_labels=True, node_color='lightgreen', 
                node_size=1500, font_weight='bold', arrows=True, 
                edge_color='green', arrowsize=20)
        plt.title('Ground Truth')
        
        plt.subplot(1, 2, 2)
        nx.draw(causal_graph, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_weight='bold', arrows=True, 
                edge_color='blue', arrowsize=20)
        plt.title('Discovered Graph')
        
        plt.tight_layout()
        plt.savefig(f"causal_discovery_{test_case['name'].replace(' ', '_')}.png")
        plt.close()
        
        # Print metrics
        print(f"Metrics for {test_case['name']}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Summarize results
    results_df = pd.DataFrame(results)
    print("\nSummary of results across all test cases:")
    print(results_df)
    
    return results_df

if __name__ == "__main__":
    print("Testing Directed Information-based Causal Discovery")
    results = test_on_multiple_structures()
    
    # Visualize overall performance
    plt.figure(figsize=(10, 6))
    results_melted = pd.melt(results, id_vars=['Test Case'], 
                             value_vars=['Precision', 'Recall', 'F1 Score', 'Lag Accuracy'])
    
    sns.barplot(x='Test Case', y='value', hue='variable', data=results_melted)
    plt.title('Performance Across Different Causal Structures')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    plt.show() 