#!/usr/bin/env python3
"""
Test script for multivariate temporal PID analysis with PAMAP2 data.
Demonstrates CVXPY and batch estimation methods for high-dimensional settings.
"""

import numpy as np
import pandas as pd
import sys
import os
from temporal_pid_multivariate import multi_lag_analysis, temporal_pid_multivariate

def generate_synthetic_multimodal_data(n_samples=1000, n_modalities=2, features_per_modality=[10, 15]):
    """Generate synthetic multimodal time series data for testing."""
    print("Generating synthetic multimodal data...")
    
    # Generate multivariate time series for each modality
    X1 = np.random.randn(n_samples, features_per_modality[0])
    X2 = np.random.randn(n_samples, features_per_modality[1])
    
    # Create target with temporal dependencies
    Y = np.zeros(n_samples)
    for t in range(3, n_samples):
        # Redundant information from both modalities
        redundant_1 = 0.3 * np.mean(X1[t-1, :3])  # First 3 features of X1
        redundant_2 = 0.3 * np.mean(X2[t-1, :3])  # First 3 features of X2
        
        # Unique information from each modality
        unique_1 = 0.2 * X1[t-2, 5]  # Specific feature from X1 with lag 2
        unique_2 = 0.2 * X2[t-3, 7]  # Specific feature from X2 with lag 3
        
        # Synergistic effect
        synergy = 0.1 * X1[t-1, 0] * X2[t-1, 0]
        
        # Combine all effects
        Y[t] = redundant_1 + redundant_2 + unique_1 + unique_2 + synergy
        
        # Add noise
        Y[t] += 0.2 * np.random.randn()
    
    # Convert to binary classification
    Y_binary = (Y > np.median(Y)).astype(int)
    
    return X1, X2, Y_binary


def test_method_comparison():
    """Compare different PID estimation methods on the same data."""
    print("\n" + "="*60)
    print("Testing Different PID Estimation Methods")
    print("="*60)
    
    # Generate test data
    X1, X2, Y = generate_synthetic_multimodal_data(
        n_samples=500,
        features_per_modality=[8, 12]
    )
    
    print(f"\nData shapes: X1={X1.shape}, X2={X2.shape}, Y={Y.shape}")
    
    # Test each method
    methods = ['joint', 'cvxpy', 'batch']
    results_by_method = {}
    
    for method in methods:
        print(f"\n{'='*40}")
        print(f"Testing {method.upper()} method")
        print(f"{'='*40}")
        
        try:
            # Set method-specific parameters
            kwargs = {
                'bins': 4,
                'dim_reduction': 'pca' if method != 'batch' else 'none',
                'n_components': 3,
                'regularization': 1e-4,  # For CVXPY
                'batch_size': 100,       # For batch method
                'n_batches': 3,          # For batch method
                'discrim_epochs': 10,    # For batch method
                'ce_epochs': 10,         # For batch method
                'seed': 42
            }
            
            # Run analysis for single lag
            result = temporal_pid_multivariate(
                X1, X2, Y, 
                lag=1,
                method=method,
                **kwargs
            )
            
            results_by_method[method] = result
            
            print(f"\nResults for lag 1:")
            print(f"  Redundancy:  {result['redundancy']:.4f}")
            print(f"  Unique X1:   {result['unique_x1']:.4f}")
            print(f"  Unique X2:   {result['unique_x2']:.4f}")
            print(f"  Synergy:     {result['synergy']:.4f}")
            print(f"  Total DI:    {result['total_di']:.4f}")
            print(f"  Sum check:   {result.get('sum_components', 0):.4f}")
            
        except Exception as e:
            print(f"Error with {method} method: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results_by_method


def test_dimensionality_scaling():
    """Test how methods scale with increasing dimensionality."""
    print("\n" + "="*60)
    print("Testing Dimensionality Scaling")
    print("="*60)
    
    dimensions = [(2, 2), (5, 5), (10, 10), (20, 20), (50, 50)]
    
    for dim1, dim2 in dimensions:
        print(f"\n{'='*40}")
        print(f"Testing with dimensions: X1={dim1}, X2={dim2}")
        print(f"{'='*40}")
        
        # Generate data with specified dimensions
        X1, X2, Y = generate_synthetic_multimodal_data(
            n_samples=300,
            features_per_modality=[dim1, dim2]
        )
        
        # Auto-select method based on dimensionality
        result = temporal_pid_multivariate(
            X1, X2, Y,
            lag=1,
            method='auto',
            bins=4,
            batch_size=50,
            n_batches=2,
            discrim_epochs=5,
            ce_epochs=5
        )
        
        print(f"Auto-selected method: {result['method']}")
        print(f"PID components: R={result['redundancy']:.3f}, "
              f"U1={result['unique_x1']:.3f}, U2={result['unique_x2']:.3f}, "
              f"S={result['synergy']:.3f}")


def test_multi_lag_analysis():
    """Test multi-lag analysis with different methods."""
    print("\n" + "="*60)
    print("Testing Multi-Lag Analysis")
    print("="*60)
    
    # Generate data
    X1, X2, Y = generate_synthetic_multimodal_data(
        n_samples=500,
        features_per_modality=[15, 20]
    )
    
    print(f"\nData shapes: X1={X1.shape}, X2={X2.shape}, Y={Y.shape}")
    
    # Test with CVXPY method
    print("\nRunning multi-lag analysis with CVXPY method...")
    results_cvxpy = multi_lag_analysis(
        X1, X2, Y,
        max_lag=3,
        bins=4,
        method='cvxpy',
        dim_reduction='pca',
        n_components=5,
        regularization=1e-4
    )
    
    print("\nCVXPY Results by lag:")
    for i, lag in enumerate(results_cvxpy['lag']):
        print(f"Lag {lag}: R={results_cvxpy['redundancy'][i]:.3f}, "
              f"U1={results_cvxpy['unique_x1'][i]:.3f}, "
              f"U2={results_cvxpy['unique_x2'][i]:.3f}, "
              f"S={results_cvxpy['synergy'][i]:.3f}")
    
    # Test with batch method
    print("\nRunning multi-lag analysis with batch method...")
    results_batch = multi_lag_analysis(
        X1, X2, Y,
        max_lag=3,
        bins=4,
        method='batch',
        batch_size=100,
        n_batches=2,
        discrim_epochs=5,
        ce_epochs=5
    )
    
    print("\nBatch Results by lag:")
    for i, lag in enumerate(results_batch['lag']):
        print(f"Lag {lag}: R={results_batch['redundancy'][i]:.3f}, "
              f"U1={results_batch['unique_x1'][i]:.3f}, "
              f"U2={results_batch['unique_x2'][i]:.3f}, "
              f"S={results_batch['synergy'][i]:.3f}")


def test_pamap_style_data():
    """Test with PAMAP-style multimodal sensor data."""
    print("\n" + "="*60)
    print("Testing with PAMAP-style Multimodal Data")
    print("="*60)
    
    # Simulate PAMAP-style data
    n_samples = 1000
    
    # Chest sensors (accelerometer, gyroscope, magnetometer)
    chest_features = 9  # 3x3 sensors
    chest_data = np.random.randn(n_samples, chest_features)
    
    # Hand sensors 
    hand_features = 9
    hand_data = np.random.randn(n_samples, hand_features)
    
    # Create activity labels based on sensor patterns
    activity = np.zeros(n_samples, dtype=int)
    for t in range(1, n_samples):
        # Activity depends on sensor patterns
        if np.mean(chest_data[t-1, :3]) > 0.5 and np.mean(hand_data[t-1, :3]) > 0.5:
            activity[t] = 1  # Walking
        elif np.mean(chest_data[t-1, 3:6]) > 0.5:
            activity[t] = 2  # Running
        else:
            activity[t] = 0  # Resting
    
    # Convert to binary (active vs resting)
    activity_binary = (activity > 0).astype(int)
    
    print(f"Modality shapes: Chest={chest_data.shape}, Hand={hand_data.shape}")
    print(f"Activity distribution: {np.bincount(activity_binary)}")
    
    # Run analysis
    print("\nRunning temporal PID analysis...")
    results = multi_lag_analysis(
        chest_data, hand_data, activity_binary,
        max_lag=5,
        bins=4,
        method='auto',  # Will auto-select based on dimensionality
        dim_reduction='clustering',  # Use clustering for sensor data
        n_components=5,
        batch_size=200,
        n_batches=3,
        discrim_epochs=10,
        ce_epochs=10
    )
    
    print("\nResults across lags:")
    print("Lag | Method | R     | U1    | U2    | S     | Total")
    print("-"*55)
    for i, lag in enumerate(results['lag']):
        method = results['method'][i] if 'method' in results else 'unknown'
        print(f"{lag:3d} | {method:6s} | "
              f"{results['redundancy'][i]:5.3f} | "
              f"{results['unique_x1'][i]:5.3f} | "
              f"{results['unique_x2'][i]:5.3f} | "
              f"{results['synergy'][i]:5.3f} | "
              f"{results['total_di'][i]:5.3f}")


if __name__ == "__main__":
    print("Multivariate Temporal PID Test Suite")
    print("====================================")
    
    # Check if required modules are available
    try:
        import cvxpy
        print("✓ CVXPY available")
    except ImportError:
        print("✗ CVXPY not available - install with: pip install cvxpy")
    
    try:
        from estimators.ce_alignment_information import CEAlignmentInformation
        print("✓ Batch estimation module available")
    except ImportError:
        print("✗ Batch estimation module not available")
    
    # Run tests
    test_method_comparison()
    test_dimensionality_scaling()
    test_multi_lag_analysis()
    test_pamap_style_data()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60) 