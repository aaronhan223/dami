import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import seaborn as sns
import os

def generate_causal_sequences(n_samples, delay=1, noise_level=0.1):
    """
    Generate two sequences X and Y where X causally influences Y with a time delay
    """
    X = np.random.randn(n_samples)
    Y = np.zeros(n_samples)
    
    # Y depends on delayed X plus noise
    for t in range(delay, n_samples):
        Y[t] = 0.7 * X[t-delay] + noise_level * np.random.randn()
    
    return X, Y

def estimate_mutual_information(X, Y, bins=10):
    """
    Estimate mutual information between X and Y
    """
    hist_2d, x_edges, y_edges = np.histogram2d(X, Y, bins=bins)
    hist_2d_prob = hist_2d / float(np.sum(hist_2d))
    
    hist_x = np.sum(hist_2d_prob, axis=1)
    hist_y = np.sum(hist_2d_prob, axis=0)
    
    H_X = entropy(hist_x)
    H_Y = entropy(hist_y)
    H_XY = entropy(hist_2d_prob.flatten())
    
    MI = H_X + H_Y - H_XY
    return MI

def estimate_directed_information(X, Y, delay_max=5, bins=10):
    """
    Estimate directed information from X to Y
    """
    DI = np.zeros(delay_max)
    
    for delay in range(1, delay_max):
        X_delayed = X[:-delay]
        Y_current = Y[delay:]
        
        DI[delay] = estimate_mutual_information(X_delayed, Y_current, bins)
    
    return DI

def run_simulation():
    n_samples = 1000
    delay = 2
    noise_levels = [0.1, 0.5, 1.0]
    
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.figure(figsize=(15, 10))
    
    for i, noise in enumerate(noise_levels):
        X, Y = generate_causal_sequences(n_samples, delay=delay, noise_level=noise)
        
        # Calculate MI and DI
        mi = estimate_mutual_information(X, Y)
        di = estimate_directed_information(X, Y)
        
        plt.subplot(2, len(noise_levels), i+1)
        plt.scatter(X, Y, alpha=0.5)
        plt.title(f'Scatter Plot (noise={noise})\nMI={mi:.3f}')
        plt.xlabel('X(t)')
        plt.ylabel('Y(t)')
        
        plt.subplot(2, len(noise_levels), i+len(noise_levels)+1)
        plt.plot(range(1, len(di)), di[1:], '-o')
        plt.title(f'Directed Information vs Time Delay\nMax DI={np.max(di):.3f}')
        plt.xlabel('Time Delay')
        plt.ylabel('Directed Information')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "di_vs_mi_comparison.png"), dpi=300)

if __name__ == "__main__":
    run_simulation()
