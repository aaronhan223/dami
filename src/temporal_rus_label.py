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
    
    # Solve with better error handling
    try:
        prob.solve(verbose=False, max_iter=50000)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem status is {prob.status}")
    except Exception as e:
        print(f"Optimization error: {e}")
        # Try with different solver
        try:
            prob.solve(verbose=False, max_iter=50000, solver=cp.ECOS)
        except:
            print("Falling back to SCS solver")
            prob.solve(verbose=False, max_iter=50000, solver=cp.SCS)

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

def create_temporal_distribution_label(X1, X2, Y, lag=1, window_size=5, bins=10):
    """
    Create a joint probability distribution from time series data at a specific temporal lag.
    
    This function creates features from data at time t-lag to predict label Y.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Univariate time series data (1D array of shape (seq_length,))
    Y : int or float
        Classification label (scalar)
    lag : int, default=1
        Temporal lag to consider (how many time steps back)
    window_size : int, default=5
        Size of temporal window ending at t-lag
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    P : numpy.ndarray
        3D array of joint probability distribution P(X1_t-lag, X2_t-lag, Y_label)
    """
    # Ensure X1 and X2 are 1D arrays
    X1 = np.asarray(X1).flatten()
    X2 = np.asarray(X2).flatten()
    
    seq_length = len(X1)
    assert len(X2) == seq_length, "X1 and X2 must have the same length"
    
    # For a single sequence, we create multiple samples by sliding windows
    # Each window position creates a sample with its corresponding label Y
    X1_windows = []
    X2_windows = []
    Y_labels = []
    
    # Generate samples by sliding window approach
    for end_pos in range(window_size, seq_length + 1):
        if lag >= end_pos:
            continue  # Skip if lag is too large for this position
            
        # Extract window ending at position (end_pos - lag)
        actual_end = end_pos - lag
        actual_start = max(0, actual_end - window_size)
        
        if actual_end > actual_start:
            X1_window = X1[actual_start:actual_end]
            X2_window = X2[actual_start:actual_end]
            
            # Use mean of window as feature
            X1_feat = np.mean(X1_window)
            X2_feat = np.mean(X2_window)
            
            X1_windows.append(X1_feat)
            X2_windows.append(X2_feat)
            Y_labels.append(Y)
    
    if len(X1_windows) == 0:
        # If no valid windows, create a minimal distribution
        P = np.zeros((bins, bins, 2))  # 2 classes as default
        P[0, 0, 0] = 1.0  # Put all probability mass in one cell
        return P
    
    X1_features = np.array(X1_windows)
    X2_features = np.array(X2_windows)
    Y_array = np.array(Y_labels)
    
    # Discretize features
    if len(np.unique(X1_features)) > 1:
        X1_edges = np.linspace(np.min(X1_features), np.max(X1_features), bins + 1)
    else:
        # If all values are the same, create uniform bins around that value
        val = X1_features[0]
        X1_edges = np.linspace(val - 1, val + 1, bins + 1)
    
    if len(np.unique(X2_features)) > 1:
        X2_edges = np.linspace(np.min(X2_features), np.max(X2_features), bins + 1)
    else:
        val = X2_features[0]
        X2_edges = np.linspace(val - 1, val + 1, bins + 1)
    
    X1_disc = np.digitize(X1_features, X1_edges) - 1
    X2_disc = np.digitize(X2_features, X2_edges) - 1
    
    # Ensure values are within bins range
    X1_disc = np.clip(X1_disc, 0, bins - 1)
    X2_disc = np.clip(X2_disc, 0, bins - 1)
    
    # For single label, we need to create a distribution that captures
    # the relationship between temporal features and the label
    # We'll use binary encoding: label present (1) or not (0)
    n_labels = 2  # Binary: label matches Y or not
    
    # Create joint probability distribution P(X1_lag, X2_lag, Y)
    P = np.zeros((bins, bins, n_labels))
    
    for i in range(len(X1_disc)):
        # All windows have the same label Y, so they all go to class 1
        P[X1_disc[i], X2_disc[i], 1] += 1
    
    # Add some probability mass to class 0 to avoid degenerate distributions
    # This represents the "background" or "no label" case
    total_samples = len(X1_disc)
    for i in range(min(total_samples, bins)):
        for j in range(min(total_samples, bins)):
            P[i, j, 0] += 0.1
    
    # Normalize to get probability distribution
    P = P / np.sum(P)
    
    return P

def temporal_pid_label_sequence(X1, X2, Y, max_lag=None, window_size=5, bins=10):
    """
    Compute temporal sequences of PID components for time series classification.
    
    This function computes how RUS quantities change over different temporal lags,
    showing which parts of the time series are most informative for classification.
    
    Parameters:
    -----------
    X1, X2 : numpy.ndarray
        Univariate time series data (1D array of shape (seq_length,))
    Y : int or float
        Classification label (scalar)
    max_lag : int, optional
        Maximum temporal lag to analyze. If None, uses seq_length - window_size
    window_size : int, default=5
        Size of temporal window for feature extraction
    bins : int, default=10
        Number of bins for discretization
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'lags': array of lag values
        - 'redundancy': array of redundancy values at each lag
        - 'unique_x1': array of unique X1 information at each lag
        - 'unique_x2': array of unique X2 information at each lag
        - 'synergy': array of synergy values at each lag
        - 'total_mi': array of total mutual information at each lag
    """
    # Ensure X1 and X2 are 1D arrays
    X1 = np.asarray(X1).flatten()
    X2 = np.asarray(X2).flatten()
    
    seq_length = len(X1)
    
    if max_lag is None:
        max_lag = seq_length - window_size
    
    results = {
        'lags': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_mi': []
    }
    
    # Compute PID for each lag
    for lag in range(max_lag + 1):
        # Create probability distribution for this lag
        P = create_temporal_distribution_label(X1, X2, Y, lag, window_size, bins)
        
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
    
    all_results = {}
    
    for class_label in [0, 1, 2]:
        print(f"\nAnalyzing Class {class_label}: {class_names[class_label]}")
        print("-"*50)
        
        # Generate synthetic data for this class
        X1, X2, Y = generate_synthetic_classification_data(
            seq_length=50, 
            class_label=class_label, 
            noise_level=0.2, 
            seed=42 + class_label  # Different seed for each class
        )
        
        print(f"Generated data: X1 shape={X1.shape}, X2 shape={X2.shape}, Y={Y}")
        
        # Plot the data
        plot_synthetic_data(
            X1, X2, Y, 
            save_path=f'../results/synthetic_class_{class_label}_data.png'
        )
        
        # Compute temporal PID sequences
        print("Computing temporal RUS sequences...")
        temporal_results = temporal_pid_label_sequence(
            X1, X2, Y,
            max_lag=30,  # Analyze up to lag 30
            window_size=10,
            bins=6
        )
        
        # Store results
        all_results[class_label] = temporal_results
        
        # Print summary statistics
        print(f"\nTemporal analysis completed for lags 0 to {len(temporal_results['lags'])-1}")
        print(f"Peak total MI: {np.max(temporal_results['total_mi']):.4f} at lag {np.argmax(temporal_results['total_mi'])}")
        print(f"Peak redundancy: {np.max(temporal_results['redundancy']):.4f} at lag {np.argmax(temporal_results['redundancy'])}")
        print(f"Peak unique X1: {np.max(temporal_results['unique_x1']):.4f} at lag {np.argmax(temporal_results['unique_x1'])}")
        print(f"Peak unique X2: {np.max(temporal_results['unique_x2']):.4f} at lag {np.argmax(temporal_results['unique_x2'])}")
        print(f"Peak synergy: {np.max(temporal_results['synergy']):.4f} at lag {np.argmax(temporal_results['synergy'])}")
        
        # Plot temporal RUS sequences for this class
        plot_temporal_rus_sequences(
            temporal_results,
            title=f'Temporal RUS - Class {class_label}: {class_names[class_label]}',
            save_path=f'../results/temporal_rus_class_{class_label}.png'
        )