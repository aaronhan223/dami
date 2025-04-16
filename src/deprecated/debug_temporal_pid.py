import numpy as np
import sys
import traceback

# Import from src directory
sys.path.append('src')
try:
    from temporal_pid import temporal_pid, generate_causal_time_series
except ImportError:
    print("Error importing temporal_pid. Make sure the src directory exists and contains temporal_pid.py.")
    sys.exit(1)

def test_minimal():
    """Minimal test of temporal_pid function."""
    try:
        print("Generating simple time series...")
        # Generate minimal data
        np.random.seed(42)
        
        # Method 1: Use built-in data generator
        X1, X2, Y = generate_causal_time_series(n_samples=100, causal_strength=0.7, noise_level=0.2, seed=42)
        print("Data generated with shape:", X1.shape)
        
        # Try with a single value of bins
        for bins in [3, 5]:
            print(f"\nTesting with bins={bins}...")
            try:
                result = temporal_pid(X1, X2, Y, lag=1, bins=bins)
                print("Result:", result)
            except Exception as e:
                print(f"Error with bins={bins}: {e}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error in test_minimal: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting minimal test...")
    test_minimal()
    print("Test complete.") 