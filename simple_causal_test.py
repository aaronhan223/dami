import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from scipy.special import rel_entr
from sklearn.metrics import mutual_info_score

# Make sure we can import from src directory
sys.path.append('src')
try:
    from temporal_pid import temporal_pid, create_probability_distribution
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
# Main Analysis
#----------------------------------------------------------------------------------

def main():
    print("Starting causal structure test script...")
    
    # Set up parameters
    n_samples = 1000  # Reduced sample size for quicker execution
    noise_level = 0.1
    lag = 1  # Just use one lag for simplicity
    bins = 8  # Fewer bins for faster computation
    seed = 42
    
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
    
    # Dictionary to store PID results
    pid_results = {}
    
    # Analyze Chain structure
    print("\nAnalyzing Chain Structure (X → Z → Y)...")
    try:
        print("Computing temporal PID...")
        start_time = time.time()
        pid_chain = temporal_pid(X_chain, Y_chain, Z_chain, lag=lag, bins=bins)
        print(f"Chain structure PID computation took {time.time() - start_time:.2f} seconds")
        
        print(f"Redundancy: {pid_chain['redundancy']:.4f}")
        print(f"Unique X: {pid_chain['unique_x1']:.4f}")
        print(f"Unique Y: {pid_chain['unique_x2']:.4f}")
        print(f"Synergy: {pid_chain['synergy']:.4f}")
        pid_results['Chain'] = pid_chain
    except Exception as e:
        print(f"Error analyzing Chain structure: {e}")
    
    # Analyze Fork structure
    print("\nAnalyzing Fork Structure (X ← Z → Y)...")
    try:
        print("Computing temporal PID...")
        start_time = time.time()
        pid_fork = temporal_pid(X_fork, Y_fork, Z_fork, lag=lag, bins=bins)
        print(f"Fork structure PID computation took {time.time() - start_time:.2f} seconds")
        
        print(f"Redundancy: {pid_fork['redundancy']:.4f}")
        print(f"Unique X: {pid_fork['unique_x1']:.4f}")
        print(f"Unique Y: {pid_fork['unique_x2']:.4f}")
        print(f"Synergy: {pid_fork['synergy']:.4f}")
        pid_results['Fork'] = pid_fork
    except Exception as e:
        print(f"Error analyzing Fork structure: {e}")
    
    # Analyze V-structure
    print("\nAnalyzing V-Structure (X → Z ← Y)...")
    try:
        print("Computing temporal PID...")
        start_time = time.time()
        pid_v = temporal_pid(X_v, Y_v, Z_v, lag=lag, bins=bins)
        print(f"V-structure PID computation took {time.time() - start_time:.2f} seconds")
        
        print(f"Redundancy: {pid_v['redundancy']:.4f}")
        print(f"Unique X: {pid_v['unique_x1']:.4f}")
        print(f"Unique Y: {pid_v['unique_x2']:.4f}")
        print(f"Synergy: {pid_v['synergy']:.4f}")
        pid_results['V-structure'] = pid_v
    except Exception as e:
        print(f"Error analyzing V-structure: {e}")
    
    # Create simple bar chart to compare RUS values across structures
    if len(pid_results) > 0:
        try:
            print("\nCreating comparison chart...")
            plt.figure(figsize=(12, 8))
            
            # Set width of bars
            barWidth = 0.25
            
            # Set positions of the bars on X axis
            r1 = np.arange(4)
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            
            # Create bars
            structures = list(pid_results.keys())
            components = ['redundancy', 'unique_x1', 'unique_x2', 'synergy']
            
            # Check if we have all 3 structures
            if len(structures) >= 3:
                plt.bar(r1, [pid_results[structures[0]][c] for c in components], 
                        width=barWidth, label=structures[0], color='blue')
                plt.bar(r2, [pid_results[structures[1]][c] for c in components], 
                        width=barWidth, label=structures[1], color='green')
                plt.bar(r3, [pid_results[structures[2]][c] for c in components], 
                        width=barWidth, label=structures[2], color='red')
            
                # Add labels and legend
                plt.xlabel('Information Component', fontsize=15)
                plt.ylabel('Information (bits)', fontsize=15)
                plt.title('PID Components Across Causal Structures', fontsize=18)
                plt.xticks([r + barWidth for r in range(4)], 
                          ['Redundancy', 'Unique X', 'Unique Y', 'Synergy'])
                plt.legend()
                
                # Save the figure
                plt.savefig('results/simple_causal_comparison.png', dpi=300, bbox_inches='tight')
                print("Figure saved to results/simple_causal_comparison.png")
            else:
                print(f"Not enough structures to plot (only have {len(structures)})")
        except Exception as e:
            print(f"Error creating plot: {e}")
    else:
        print("No results to plot")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 