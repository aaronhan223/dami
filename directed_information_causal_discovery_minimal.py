#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Directed Information-based Causal Discovery with Time-Lagged Relationships

This script demonstrates the core concepts of using directed information for causal discovery
with a focus on simplicity and minimal computational requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)

def generate_time_series(causal_structure, time_lags, noise_level=0.1, length=500):
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
    var_map : dict
        Mapping from variable names to indices
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

def correlation_coefficient(x, y):
    """
    Calculate linear correlation coefficient as a simple measure for relationship strength.
    """
    return np.corrcoef(x, y)[0, 1]

def time_lagged_correlation(x, y, max_lag=5):
    """
    Calculate time-lagged correlation from X to Y.
    
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
    max_corr : float
        Maximum correlation value
    optimal_lag : int
        Lag that maximizes correlation
    """
    n = len(x)
    corr_values = []
    
    for lag in range(1, min(max_lag + 1, n // 4)):
        x_past = x[:-lag]
        y_current = y[lag:]
        
        # Only use part of the data that aligns given the lag
        valid_length = min(len(x_past), len(y_current))
        x_past = x_past[-valid_length:]
        y_current = y_current[-valid_length:]
        
        # Calculate correlation (simpler than directed information)
        corr = correlation_coefficient(x_past, y_current)
        corr_values.append(corr)
    
    if not corr_values:
        return 0, 0
    
    # Find the lag that maximizes correlation
    optimal_lag = np.argmax(np.abs(corr_values)) + 1
    max_corr = corr_values[optimal_lag - 1]
    
    return max_corr, optimal_lag

def discover_causal_structure(data, var_names=None, max_lag=5, threshold=0.1):
    """
    Discover causal structure using time-lagged correlation as a simple proxy for directed information.
    
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
    correlation_values : dict
        Dictionary of correlation values
    """
    n_vars = data.shape[1]
    
    if var_names is None:
        var_names = [f'X{i}' for i in range(n_vars)]
    
    # Initialize results
    causal_graph = nx.DiGraph()
    for var in var_names:
        causal_graph.add_node(var)
    
    discovered_lags = {}
    correlation_values = {}
    
    # Check all possible pairs
    print("Computing time-lagged correlations...")
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:  # Don't check self-causality
                print(f"Processing {var_names[i]} → {var_names[j]}", end='\r')
                corr_ij, lag_ij = time_lagged_correlation(data[:, i], data[:, j], max_lag)
                
                correlation_values[(var_names[i], var_names[j])] = corr_ij
                
                # Check if the relationship is significant
                if abs(corr_ij) > threshold:
                    causal_graph.add_edge(var_names[i], var_names[j], weight=abs(corr_ij))
                    discovered_lags[(var_names[i], var_names[j])] = lag_ij
    
    print("\nDone computing time-lagged correlations.")
    return causal_graph, discovered_lags, correlation_values

def evaluate_performance(true_structure, discovered_structure, time_lags, discovered_lags):
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

def plot_causal_graphs(true_graph, discovered_graph, discovered_lags, save_path=None):
    """
    Plot the true and discovered causal graphs side by side.
    """
    plt.figure(figsize=(12, 5))
    
    # Same layout for both graphs
    pos = nx.spring_layout(true_graph, seed=42)
    
    # Plot true graph
    plt.subplot(1, 2, 1)
    nx.draw(true_graph, pos, with_labels=True, node_color='lightgreen', 
            node_size=1500, font_weight='bold', arrows=True, 
            edge_color='green', arrowsize=20)
    plt.title('Ground Truth')
    
    # Plot discovered graph
    plt.subplot(1, 2, 2)
    nx.draw(discovered_graph, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_weight='bold', arrows=True, 
            edge_color='blue', arrowsize=20)
    
    # Add lag information
    edge_labels = {(u, v): f"lag={discovered_lags.get((u, v), '?')}" 
                  for u, v in discovered_graph.edges()}
    nx.draw_networkx_edge_labels(discovered_graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title('Discovered Graph')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()

def test_simple_chain():
    """
    Test the causal discovery method on a simple chain X → Y → Z.
    """
    print("\nTesting on Simple Chain X → Y → Z")
    
    # Define causal structure
    structure = {('X', 'Y'): 0.8, ('Y', 'Z'): 0.7}
    time_lags = {('X', 'Y'): 2, ('Y', 'Z'): 1}
    
    # Generate data
    data, var_map = generate_time_series(structure, time_lags, noise_level=0.1, length=500)
    
    # Discover causal structure
    var_names = sorted(var_map.keys())
    causal_graph, discovered_lags, corr_values = discover_causal_structure(
        data, var_names, max_lag=5, threshold=0.1
    )
    
    # Create true graph for visualization
    true_graph = nx.DiGraph()
    for var in var_map.keys():
        true_graph.add_node(var)
    for (i, j), strength in structure.items():
        true_graph.add_edge(i, j, weight=strength)
    
    # Evaluate performance
    metrics = evaluate_performance(structure, causal_graph, time_lags, discovered_lags)
    
    # Print results
    print("\nCorrelation values:")
    for (from_var, to_var), corr in corr_values.items():
        print(f"{from_var} → {to_var}: {corr:.4f}")
    
    print("\nDiscovered lags:")
    for (from_var, to_var), lag in discovered_lags.items():
        print(f"{from_var} → {to_var}: {lag}")
    
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot graphs
    plt.figure(figsize=(12, 5))
    plot_causal_graphs(true_graph, causal_graph, discovered_lags, "simple_chain_result.png")
    plt.close()
    
    return metrics

def test_common_cause():
    """
    Test the causal discovery method on a common cause structure X → Y, X → Z.
    """
    print("\nTesting on Common Cause X → Y, X → Z")
    
    # Define causal structure
    structure = {('X', 'Y'): 0.8, ('X', 'Z'): 0.7}
    time_lags = {('X', 'Y'): 1, ('X', 'Z'): 3}
    
    # Generate data
    data, var_map = generate_time_series(structure, time_lags, noise_level=0.1, length=500)
    
    # Discover causal structure
    var_names = sorted(var_map.keys())
    causal_graph, discovered_lags, corr_values = discover_causal_structure(
        data, var_names, max_lag=5, threshold=0.1
    )
    
    # Create true graph for visualization
    true_graph = nx.DiGraph()
    for var in var_map.keys():
        true_graph.add_node(var)
    for (i, j), strength in structure.items():
        true_graph.add_edge(i, j, weight=strength)
    
    # Evaluate performance
    metrics = evaluate_performance(structure, causal_graph, time_lags, discovered_lags)
    
    # Print results
    print("\nCorrelation values:")
    for (from_var, to_var), corr in corr_values.items():
        print(f"{from_var} → {to_var}: {corr:.4f}")
    
    print("\nDiscovered lags:")
    for (from_var, to_var), lag in discovered_lags.items():
        print(f"{from_var} → {to_var}: {lag}")
    
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot graphs
    plt.figure(figsize=(12, 5))
    plot_causal_graphs(true_graph, causal_graph, discovered_lags, "common_cause_result.png")
    plt.close()
    
    return metrics

def test_cycle():
    """
    Test the causal discovery method on a cyclic structure X → Y → Z → X.
    """
    print("\nTesting on Cycle X → Y → Z → X")
    
    # Define causal structure
    structure = {('X', 'Y'): 0.6, ('Y', 'Z'): 0.7, ('Z', 'X'): 0.5}
    time_lags = {('X', 'Y'): 2, ('Y', 'Z'): 1, ('Z', 'X'): 3}
    
    # Generate data
    data, var_map = generate_time_series(structure, time_lags, noise_level=0.1, length=500)
    
    # Discover causal structure
    var_names = sorted(var_map.keys())
    causal_graph, discovered_lags, corr_values = discover_causal_structure(
        data, var_names, max_lag=5, threshold=0.1
    )
    
    # Create true graph for visualization
    true_graph = nx.DiGraph()
    for var in var_map.keys():
        true_graph.add_node(var)
    for (i, j), strength in structure.items():
        true_graph.add_edge(i, j, weight=strength)
    
    # Evaluate performance
    metrics = evaluate_performance(structure, causal_graph, time_lags, discovered_lags)
    
    # Print results
    print("\nCorrelation values:")
    for (from_var, to_var), corr in corr_values.items():
        print(f"{from_var} → {to_var}: {corr:.4f}")
    
    print("\nDiscovered lags:")
    for (from_var, to_var), lag in discovered_lags.items():
        print(f"{from_var} → {to_var}: {lag}")
    
    print("\nPerformance metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot graphs
    plt.figure(figsize=(12, 5))
    plot_causal_graphs(true_graph, causal_graph, discovered_lags, "cycle_result.png")
    plt.close()
    
    return metrics

def summarize_results(results_dict):
    """
    Summarize results across all test cases.
    """
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    # Calculate average metrics
    avg_precision = np.mean([metrics['Precision'] for metrics in results_dict.values()])
    avg_recall = np.mean([metrics['Recall'] for metrics in results_dict.values()])
    avg_f1 = np.mean([metrics['F1 Score'] for metrics in results_dict.values()])
    avg_lag_acc = np.mean([metrics['Lag Accuracy'] for metrics in results_dict.values()])
    
    # Print each test case result
    for test_name, metrics in results_dict.items():
        print(f"\n{test_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Print average results
    print("\nAVERAGE ACROSS ALL TEST CASES:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    print(f"  Lag Accuracy: {avg_lag_acc:.4f}")
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    
    # Set up data for plotting
    test_names = list(results_dict.keys())
    precision_values = [metrics['Precision'] for metrics in results_dict.values()]
    recall_values = [metrics['Recall'] for metrics in results_dict.values()]
    f1_values = [metrics['F1 Score'] for metrics in results_dict.values()]
    lag_acc_values = [metrics['Lag Accuracy'] for metrics in results_dict.values()]
    
    # Create bar positions
    bar_width = 0.2
    r1 = np.arange(len(test_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Create grouped bar chart
    plt.bar(r1, precision_values, width=bar_width, label='Precision', color='blue')
    plt.bar(r2, recall_values, width=bar_width, label='Recall', color='green')
    plt.bar(r3, f1_values, width=bar_width, label='F1 Score', color='red')
    plt.bar(r4, lag_acc_values, width=bar_width, label='Lag Accuracy', color='orange')
    
    # Add labels and title
    plt.xlabel('Test Case')
    plt.ylabel('Score')
    plt.title('Performance Across Different Causal Structures')
    plt.xticks([r + bar_width*1.5 for r in range(len(test_names))], test_names)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Testing Time-Lagged Correlation-based Causal Discovery")
    
    # Run all tests
    results = {}
    results["Simple Chain"] = test_simple_chain()
    results["Common Cause"] = test_common_cause()
    results["Cycle"] = test_cycle()
    
    # Summarize results
    summarize_results(results)
    
    print("\nAll tests completed. Results saved to disk.") 