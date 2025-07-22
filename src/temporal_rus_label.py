import numpy as np
import cvxpy as cp
from scipy.special import rel_entr
import matplotlib.pyplot as plt
import os
import pdb


def MI(P: np.ndarray):
    """
    Calculate mutual information from a 2D joint probability distribution.
    """
    margin_1 = P.sum(axis=1)
    margin_2 = P.sum(axis=0)
    outer = np.outer(margin_1, margin_2)
    
    # Calculate KL divergence
    return np.sum(rel_entr(P, outer))


def solve_Q_label(P: np.ndarray):
    """
    Compute optimal Q given 3D array P for classification setting.
    
    This function solves an optimization problem to find the distribution Q
    that preserves marginals P(X1,Y) and P(X2,Y) while minimizing I(X1;X2|Y).
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution P(X1_features, X2_features, Y_label)
        
    Returns:
    --------
    Q : numpy.ndarray
        Optimized joint distribution with minimal synergy
    """
    # Compute marginals
    Py = P.sum(axis=0).sum(axis=0)
    Px1 = P.sum(axis=1).sum(axis=1)
    Px2 = P.sum(axis=0).sum(axis=1)
    Px2y = P.sum(axis=0)
    Px1y = P.sum(axis=1)
    
    # Define optimization variables
    Q = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]
    Q_x1x2 = [cp.Variable((P.shape[0], P.shape[1]), nonneg=True) for i in range(P.shape[2])]

    # Constraints that conditional distributions sum to 1
    sum_to_one_Q = cp.sum([cp.sum(q) for q in Q]) == 1

    # [A]: p(x1, y) == q(x1, y) constraints
    A_cstrs = []
    for x1 in range(P.shape[0]):
        for y in range(P.shape[2]):
            vars = []
            for x2 in range(P.shape[1]):
                vars.append(Q[y][x1, x2])
            A_cstrs.append(cp.sum(vars) == Px1y[x1,y])
    
    # [B]: p(x2, y) == q(x2, y) constraints
    B_cstrs = []
    for x2 in range(P.shape[1]):
        for y in range(P.shape[2]):
            vars = []
            for x1 in range(P.shape[0]):
                vars.append(Q[y][x1, x2])
            B_cstrs.append(cp.sum(vars) == Px2y[x2,y])

    # KL divergence - Product distribution constraints
    Q_pdt_dist_cstrs = [cp.sum(Q) / P.shape[2] == Q_x1x2[i] for i in range(P.shape[2])]

    # Objective: minimize I(X1; X2 | Y)
    obj = cp.sum([cp.sum(cp.rel_entr(Q[i], Q_x1x2[i])) for i in range(P.shape[2])])
    all_constrs = [sum_to_one_Q] + A_cstrs + B_cstrs + Q_pdt_dist_cstrs
    prob = cp.Problem(cp.Minimize(obj), all_constrs)
    
    # Solve with comprehensive error handling - try multiple solvers
    solvers_to_try = [
        (None, "CLARABEL (default)"),  # Default solver first
        (cp.ECOS, "ECOS"),
        (cp.SCS, "SCS"),
        (cp.OSQP, "OSQP"),
        (cp.CVXOPT, "CVXOPT")
    ]
    
    solved = False
    for solver, solver_name in solvers_to_try:
        try:
            if solver is None:
                prob.solve(verbose=False, max_iter=50000)
            else:
                prob.solve(verbose=False, max_iter=50000, solver=solver)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                solved = True
                break
            elif prob.status in ["infeasible", "unbounded"]:
                print(f"Warning: Problem is {prob.status} with {solver_name}")
                continue
            else:
                print(f"Warning: Problem status is {prob.status} with {solver_name}")
                continue
                
        except Exception as e:
            print(f"Solver {solver_name} failed: {e}")
            continue
    
    if not solved:
        print("Warning: All solvers failed. Trying SCS with relaxed parameters...")
        try:
            prob.solve(verbose=True, max_iter=100000, solver=cp.SCS, eps=1e-6)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Final attempt status: {prob.status}")
        except Exception as e:
            print(f"Final solver attempt failed: {e}")
            # Return a fallback solution - uniform distribution
            print("Returning uniform distribution as fallback")
            uniform_q = np.ones(Q[0].shape) / np.prod(Q[0].shape)
            return np.stack([uniform_q for _ in range(len(Q))], axis=2)

    # Convert to numpy array
    return np.stack([q.value for q in Q], axis=2)


def CoI_label(P: np.ndarray):
    """
    Calculate co-information (redundancy) from classification distribution.
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution P(X1_features, X2_features, Y_label)
        
    Returns:
    --------
    redundancy : float
        Redundant information between X1_features and X2_features about Y_label
    """
    # MI(Y; X1)
    A = P.sum(axis=1)

    # MI(Y; X2)
    B = P.sum(axis=0)

    # MI(Y; (X1, X2))
    C = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
    
    # I(Y; X1; X2)
    return MI(A) + MI(B) - MI(C)


def UI_label(P, cond_id=0):
    """
    Calculate unique information from classification distribution.
    
    Parameters:
    -----------
    P : numpy.ndarray
        3D joint probability distribution P(X1_features, X2_features, Y_label)
    cond_id : int, default=0
        If 0, calculate unique information of X2; if 1, calculate unique information of X1
        
    Returns:
    --------
    unique_info : float
        Unique information from one source about Y_label
    """
    sum_val = 0.0

    if cond_id == 0:
        # Unique info from X2 (condition on X1)
        J = P.sum(axis=(1, 2))  # marginal of X1
        for i in range(P.shape[0]):
            P_slice = P[i,:,:]
            if np.sum(P_slice) > 0:  # Avoid division by zero
                sum_val += MI(P_slice/np.sum(P_slice)) * J[i]
    elif cond_id == 1:
        # Unique info from X1 (condition on X2)
        J = P.sum(axis=(0, 2))  # marginal of X2
        for i in range(P.shape[1]):
            P_slice = P[:,i,:]
            if np.sum(P_slice) > 0:  # Avoid division by zero
                sum_val += MI(P_slice/np.sum(P_slice)) * J[i]
    else:
        raise ValueError("cond_id must be 0 or 1")

    return sum_val


def CI_label(P, Q):
    """
    Calculate synergistic information from classification distributions.
    
    Parameters:
    -----------
    P : numpy.ndarray
        Original 3D joint probability distribution P(X1_features, X2_features, Y_label)
    Q : numpy.ndarray
        Optimized 3D joint distribution with minimal synergy
        
    Returns:
    --------
    synergy : float
        Synergistic information from X1_features and X2_features about Y_label
    """
    # Ensure P and Q have the same shape
    assert P.shape == Q.shape
    
    # Reshape to 2D for mutual information calculation
    P_ = P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1]))
    Q_ = Q.transpose([2, 0, 1]).reshape((Q.shape[2], Q.shape[0]*Q.shape[1]))
    
    # Calculate total MI in P minus total MI in Q (synergy)
    return MI(P_) - MI(Q_)


def create_temporal_distribution_label_multi(X1_list, X2_list, Y_list, n_labels=None, lag=1, bins=10):
    """
    Create a joint probability distribution from multiple time series with different labels.
    
    This function processes multiple sequences, each with its own label, and creates
    a joint distribution P(X1_past, X2_past, Y_label) using data at lag positions.
    
    Parameters:
    -----------
    X1_list, X2_list : list of numpy.ndarray
        List of time series, one per sequence
    Y_list : list or numpy.ndarray
        Classification labels for each sequence
    n_labels : int, optional
        Expected number of distinct labels. If None, inferred from Y_list
    lag : int, default=1
        Temporal lag to consider (how many time steps back)
    bins : int, default=10
        Number of bins for discretization of X1 and X2
        
    Returns:
    --------
    P : numpy.ndarray
        3D array of joint probability distribution P(X1_past, X2_past, Y_label)
    """
    # Ensure inputs are lists
    if not isinstance(X1_list, list):
        X1_list = [X1_list]
    if not isinstance(X2_list, list):
        X2_list = [X2_list]
    if not isinstance(Y_list, (list, np.ndarray)):
        Y_list = [Y_list]
    
    # Convert Y_list to numpy array for easier processing
    Y_array = np.array(Y_list)
    
    # Check for distinct labels
    unique_labels = np.unique(Y_array)
    actual_n_labels = len(unique_labels)
    
    if n_labels is None:
        n_labels = actual_n_labels
    else:
        # Ensure n_labels is at least as large as actual number of unique labels
        n_labels = max(n_labels, actual_n_labels)
    
    # Warn if we have a degenerate case (only one label)
    if actual_n_labels == 1:
        print(f"Warning: Only one unique label found ({unique_labels[0]}). "
              f"This may lead to degenerate distributions. Consider providing multiple classes.")
    
    # Collect all data points
    all_X1_data = []
    all_X2_data = []
    all_Y_labels = []
    
    # Process each sequence
    for X1, X2, Y in zip(X1_list, X2_list, Y_list):
        X1 = np.asarray(X1).flatten()
        X2 = np.asarray(X2).flatten()
        
        if len(X1) != len(X2):
            raise ValueError("X1 and X2 must have the same length")
        
        # For each sequence, extract data at lag position
        if lag == 0:
            X1_past = X1
            X2_past = X2
        else:
            X1_past = X1[:-lag]
            X2_past = X2[:-lag]
        
        # Skip if no data after lag
        if len(X1_past) == 0:
            continue
            
        # Add data from this sequence
        all_X1_data.extend(X1_past)
        all_X2_data.extend(X2_past)
        # Each point is associated with the sequence's label
        all_Y_labels.extend([Y] * len(X1_past))
    
    if len(all_X1_data) == 0:
        # Create a minimal uniform distribution if no data
        P = np.ones((bins, bins, n_labels)) / (bins * bins * n_labels)
        return P
    
    # Convert to arrays
    X1_data = np.array(all_X1_data)
    X2_data = np.array(all_X2_data)
    Y_data = np.array(all_Y_labels)
    
    # Discretize X1 and X2 data (same logic as temporal_pid.py)
    if not np.array_equal(X1_data, X1_data.astype(int)) or not np.array_equal(X2_data, X2_data.astype(int)):
        # Continuous data
        x1_edges = np.linspace(np.min(X1_data), np.max(X1_data), bins + 1)
        x2_edges = np.linspace(np.min(X2_data), np.max(X2_data), bins + 1)
        
        x1_bins = np.digitize(X1_data, x1_edges) - 1
        x2_bins = np.digitize(X2_data, x2_edges) - 1
        
        x1_bins = np.clip(x1_bins, 0, bins - 1)
        x2_bins = np.clip(x2_bins, 0, bins - 1)
    else:
        # Already discrete
        x1_bins = X1_data
        x2_bins = X2_data
        
        # Remap to contiguous integers
        x1_unique = np.unique(x1_bins)
        x2_unique = np.unique(x2_bins)
        
        x1_map = {val: i for i, val in enumerate(x1_unique)}
        x2_map = {val: i for i, val in enumerate(x2_unique)}
        
        x1_bins = np.array([x1_map[val] for val in x1_bins])
        x2_bins = np.array([x2_map[val] for val in x2_bins])
        
        bins = max(len(x1_unique), len(x2_unique))
    
    # Handle Y labels (already discrete)
    # Create label mapping to ensure contiguous indices starting from 0
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    y_bins = np.array([label_map[label] for label in Y_data])
    
    # Create joint probability distribution
    P = np.zeros((bins, bins, n_labels))
    
    # Build histogram
    for i in range(len(x1_bins)):
        P[x1_bins[i], x2_bins[i], y_bins[i]] += 1
    
    # Add small pseudocount to avoid zero probabilities (optional, for numerical stability)
    epsilon = 1e-10
    P = P + epsilon
    
    # Normalize
    P = P / np.sum(P)
    
    return P


def plot_temporal_rus_sequences(results, title=None, save_path=None):
    """
    Plot temporal RUS sequences showing how information components change over time.
    
    Parameters:
    -----------
    results : dict
        Output from temporal_pid_label_sequence
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(14, 10))
    
    # Plot 1: All components over lags
    plt.subplot(2, 2, 1)
    plt.plot(results['lags'], results['total_mi'], 'k-', linewidth=2, label='Total MI')
    plt.plot(results['lags'], results['redundancy'], 'b.-', label='Redundancy', markersize=8)
    plt.plot(results['lags'], results['unique_x1'], 'g.-', label='Unique X1', markersize=8)
    plt.plot(results['lags'], results['unique_x2'], 'r.-', label='Unique X2', markersize=8)
    plt.plot(results['lags'], results['synergy'], 'm.-', label='Synergy', markersize=8)
    plt.xlabel('Temporal Lag', fontsize=12)
    plt.ylabel('Information (bits)', fontsize=12)
    plt.title('PID Components vs Temporal Lag', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Stacked area plot
    plt.subplot(2, 2, 2)
    plt.stackplot(
        results['lags'], 
        results['redundancy'], 
        results['unique_x1'], 
        results['unique_x2'], 
        results['synergy'],
        labels=['Redundancy', 'Unique X1', 'Unique X2', 'Synergy'],
        colors=['blue', 'green', 'red', 'magenta'],
        alpha=0.7
    )
    plt.plot(results['lags'], results['total_mi'], 'k--', linewidth=2, label='Total MI')
    plt.xlabel('Temporal Lag', fontsize=12)
    plt.ylabel('Information (bits)', fontsize=12)
    plt.title('Stacked PID Components', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Normalized components
    plt.subplot(2, 2, 3)
    total_nonzero = np.where(results['total_mi'] > 0, results['total_mi'], 1)
    plt.plot(results['lags'], results['redundancy'] / total_nonzero * 100, 'b.-', label='Redundancy %', markersize=8)
    plt.plot(results['lags'], results['unique_x1'] / total_nonzero * 100, 'g.-', label='Unique X1 %', markersize=8)
    plt.plot(results['lags'], results['unique_x2'] / total_nonzero * 100, 'r.-', label='Unique X2 %', markersize=8)
    plt.plot(results['lags'], results['synergy'] / total_nonzero * 100, 'm.-', label='Synergy %', markersize=8)
    plt.xlabel('Temporal Lag', fontsize=12)
    plt.ylabel('Percentage of Total MI', fontsize=12)
    plt.title('Normalized PID Components', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Plot 4: Unique information ratio
    plt.subplot(2, 2, 4)
    unique_x1_safe = np.where(results['unique_x1'] > 0, results['unique_x1'], 1e-10)
    unique_x2_safe = np.where(results['unique_x2'] > 0, results['unique_x2'], 1e-10)
    ratio = unique_x1_safe / unique_x2_safe
    plt.plot(results['lags'], ratio, 'ko-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Temporal Lag', fontsize=12)
    plt.ylabel('Unique X1 / Unique X2', fontsize=12)
    plt.title('Unique Information Ratio', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def generate_synthetic_classification_data(seq_length=50, class_label=0, noise_level=0.1, seed=None):
    """
    Generate synthetic time series data with a classification label.
    
    Parameters:
    -----------
    seq_length : int, default=50
        Length of the time series
    class_label : int, default=0
        Classification label (0, 1, or 2)
    noise_level : float, default=0.1
        Amount of noise to add
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X1, X2 : numpy.ndarray
        Univariate time series data (shape: (seq_length,))
    Y : int
        Class label
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 1, seq_length)
    
    if class_label == 0:
        # Class 0: X1 has increasing trend, X2 has high frequency oscillations
        X1 = 2 * t + noise_level * np.random.randn(seq_length)
        X2 = np.sin(4 * np.pi * t) + noise_level * np.random.randn(seq_length)
        
    elif class_label == 1:
        # Class 1: X1 has decreasing trend, X2 has low frequency oscillations
        X1 = 2 * (1 - t) + noise_level * np.random.randn(seq_length)
        X2 = np.sin(np.pi * t) + noise_level * np.random.randn(seq_length)
        
    elif class_label == 2:
        # Class 2: Both X1 and X2 needed together (synergistic pattern)
        # X1 has step function, X2 has complementary step function
        step_point = seq_length // 2
        X1 = np.zeros(seq_length)
        X2 = np.zeros(seq_length)
        X1[:step_point] = 1
        X2[step_point:] = 1
        X1 += noise_level * np.random.randn(seq_length)
        X2 += noise_level * np.random.randn(seq_length)
    else:
        raise ValueError(f"Invalid class_label: {class_label}. Must be 0, 1, or 2.")
    
    return X1, X2, class_label


def plot_synthetic_data(X1, X2, Y, save_path=None):
    """
    Plot the synthetic time series data.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Univariate time series data (shape: (seq_length,))
    Y : int
        Class label
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 4))
    
    t = np.arange(len(X1))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, X1, 'b-', label='X1', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'X1 - Class {Y}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(t, X2, 'r-', label='X2', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'X2 - Class {Y}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def temporal_pid_label_multi_sequence(X1_list, X2_list, Y_list, max_lag=None, bins=10):
    """
    Compute temporal PID components for multiple time series with different class labels.
    
    This function analyzes how information components change over different temporal lags
    when considering multiple sequences from different classes together.
    
    Parameters:
    -----------
    X1_list, X2_list : list of numpy.ndarray
        Lists of time series, one per sequence
    Y_list : list or numpy.ndarray
        Classification labels for each sequence
    max_lag : int, optional
        Maximum temporal lag to analyze. If None, uses minimum sequence length - 1
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    results : dict
        Dictionary containing PID components at each lag
    """
    # Ensure inputs are lists
    if not isinstance(X1_list, list):
        X1_list = [X1_list]
    if not isinstance(X2_list, list):
        X2_list = [X2_list]
    if not isinstance(Y_list, (list, np.ndarray)):
        Y_list = [Y_list]
    
    # Check that all lists have the same length
    if not (len(X1_list) == len(X2_list) == len(Y_list)):
        raise ValueError("X1_list, X2_list, and Y_list must have the same length")
    
    # Find minimum sequence length if max_lag not specified
    if max_lag is None:
        min_length = min(len(X1) for X1 in X1_list)
        max_lag = min_length - 1
    
    # Get unique labels
    unique_labels = np.unique(Y_list)
    n_labels = len(unique_labels)
    
    if n_labels < 2:
        print(f"Warning: Only {n_labels} unique label(s) found. "
              f"PID analysis works best with multiple classes.")
    
    results = {
        'lags': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_mi': [],
        'n_sequences': len(X1_list),
        'n_labels': n_labels,
        'unique_labels': unique_labels.tolist()
    }
    
    # Compute PID for each lag
    for lag in range(max_lag + 1):
        # Create probability distribution for this lag using all sequences
        P = create_temporal_distribution_label_multi(X1_list, X2_list, Y_list, n_labels, lag, bins)
        
        # Check if distribution is valid (not all zeros)
        if np.sum(P) == 0:
            print(f"Warning: Empty distribution at lag {lag}. Skipping.")
            continue
        
        # Optimize to get Q
        Q = solve_Q_label(P)
        
        # Calculate PID components
        redundancy = CoI_label(Q)
        unique_x1 = UI_label(Q, cond_id=1)
        unique_x2 = UI_label(Q, cond_id=0)
        synergy = CI_label(P, Q)
        
        # Calculate total mutual information
        total_mi = MI(P.transpose([2, 0, 1]).reshape((P.shape[2], P.shape[0]*P.shape[1])))
        
        # Store results
        results['lags'].append(lag)
        results['redundancy'].append(redundancy)
        results['unique_x1'].append(unique_x1)
        results['unique_x2'].append(unique_x2)
        results['synergy'].append(synergy)
        results['total_mi'].append(total_mi)
    
    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    return results

# Example usage and synthetic experiment
if __name__ == "__main__":
    print("Temporal RUS Analysis for Time Series Classification")
    print("="*60)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('../results'):
        os.makedirs('../results')
    
    # Test with different class labels
    class_names = {0: "Increasing trend + High freq", 
                   1: "Decreasing trend + Low freq",
                   2: "Synergistic step functions"}
    
    # Generate synthetic data for all classes
    print("\nGenerating synthetic data for all classes...")
    X1_list = []
    X2_list = []
    Y_list = []
    
    # Generate multiple sequences per class for better statistics
    n_sequences_per_class = 4
    
    for class_label in [0, 1, 2]:
        print(f"Generating {n_sequences_per_class} sequences for Class {class_label}: {class_names[class_label]}")
        
        for seq_idx in range(n_sequences_per_class):
            # Generate synthetic data for this class with different random seeds
            X1, X2, Y = generate_synthetic_classification_data(
                seq_length=50, 
                class_label=class_label, 
                noise_level=0.2, 
                seed=42 + class_label * 10 + seq_idx  # Different seed for each sequence
            )
            
            X1_list.append(X1)
            X2_list.append(X2)
            Y_list.append(Y)
            
            # Plot first sequence of each class
            if seq_idx == 0:
                plot_synthetic_data(
                    X1, X2, Y, 
                    save_path=f'../results/synthetic_class_{class_label}_data.png'
                )
    
    print(f"\nTotal sequences generated: {len(X1_list)}")
    print(f"Unique labels: {np.unique(Y_list)}")
    print(f"Sequences per class: {n_sequences_per_class}")
    
    # Analyze all sequences together
    print("\n" + "="*60)
    print("Analyzing all sequences together with temporal_pid_label_multi_sequence")
    print("="*60)
    
    # Compute temporal PID for all sequences together
    temporal_results = temporal_pid_label_multi_sequence(
        X1_list, X2_list, Y_list,
        max_lag=30,  # Analyze up to lag 30
        bins=6
    )
    
    # Print summary statistics
    print(f"\nTemporal analysis completed for lags 0 to {len(temporal_results['lags'])-1}")
    print(f"Number of sequences analyzed: {temporal_results['n_sequences']}")
    print(f"Number of unique labels: {temporal_results['n_labels']}")
    print(f"Unique labels: {temporal_results['unique_labels']}")
    
    print(f"\nPeak total MI: {np.max(temporal_results['total_mi']):.4f} at lag {np.argmax(temporal_results['total_mi'])}")
    print(f"Peak redundancy: {np.max(temporal_results['redundancy']):.4f} at lag {np.argmax(temporal_results['redundancy'])}")
    print(f"Peak unique X1: {np.max(temporal_results['unique_x1']):.4f} at lag {np.argmax(temporal_results['unique_x1'])}")
    print(f"Peak unique X2: {np.max(temporal_results['unique_x2']):.4f} at lag {np.argmax(temporal_results['unique_x2'])}")
    print(f"Peak synergy: {np.max(temporal_results['synergy']):.4f} at lag {np.argmax(temporal_results['synergy'])}")
    
    # Plot temporal RUS sequences
    plot_temporal_rus_sequences(
        temporal_results,
        title='Temporal RUS Analysis - All Classes Combined',
        save_path='../results/temporal_rus_all_classes.png'
    )
