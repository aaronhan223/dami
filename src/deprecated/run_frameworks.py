import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Import implementations
from multi_lag_directed_info_impl import (
    generate_chain_structure,
    generate_fork_structure,
    generate_v_structure,
    test_multi_lag_pid_framework
)

from enhanced_causal_pid_impl import (
    test_and_compare_causal_structures
)

# Import visualizations
from multi_lag_directed_info_viz import (
    demo_multi_lag_framework
)

from enhanced_causal_pid_viz import (
    demo_enhanced_causal_framework
)

def main():
    """
    Run both frameworks and demonstrate their capabilities.
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("=" * 80)
    print(" Multi-Lag Directed Information Framework and Enhanced Causal PID Framework Demo")
    print("=" * 80)
    
    # Ask the user which framework to run
    print("\nWhich framework would you like to run?")
    print("1. Multi-Lag Directed Information Framework")
    print("2. Enhanced Causal PID Framework")
    print("3. Both frameworks")
    print("4. Run tests for both frameworks")
    
    choice = input("\nEnter your choice (1-4): ")
    
    start_time = time.time()
    
    if choice == '1':
        print("\nRunning Multi-Lag Directed Information Framework...\n")
        results = demo_multi_lag_framework()
        print("\nMulti-Lag Directed Information Framework completed.")
        
    elif choice == '2':
        print("\nRunning Enhanced Causal PID Framework...\n")
        results = demo_enhanced_causal_framework()
        print("\nEnhanced Causal PID Framework completed.")
        
    elif choice == '3':
        print("\nRunning both frameworks...\n")
        
        print("\n1. Multi-Lag Directed Information Framework")
        print("-" * 50)
        multi_lag_results = demo_multi_lag_framework()
        
        print("\n2. Enhanced Causal PID Framework")
        print("-" * 50)
        enhanced_results = demo_enhanced_causal_framework()
        
        results = {
            'multi_lag': multi_lag_results,
            'enhanced': enhanced_results
        }
        
        print("\nBoth frameworks completed.")
        
    elif choice == '4':
        print("\nRunning tests for both frameworks...\n")
        
        print("\n1. Testing Multi-Lag Directed Information Framework")
        print("-" * 50)
        multi_lag_test = test_multi_lag_pid_framework()
        
        print("Multi-Lag Framework Test Results:")
        print(f"  Chain Structure Detection: {'Correct' if multi_lag_test['chain_structure']['correct'] else 'Incorrect'}")
        print(f"  Fork Structure Detection: {'Correct' if multi_lag_test['fork_structure']['correct'] else 'Incorrect'}")
        print(f"  V-Structure Detection: {'Correct' if multi_lag_test['v_structure']['correct'] else 'Incorrect'}")
        print(f"  Overall Accuracy: {multi_lag_test['overall_accuracy']:.2f}")
        
        print("\n2. Testing Enhanced Causal PID Framework")
        print("-" * 50)
        enhanced_test = test_and_compare_causal_structures()
        
        print("Enhanced Causal PID Framework Test Results:")
        print(f"  Chain Structure Detection: {'Correct' if enhanced_test['chain_structure']['correct'] else 'Incorrect'}")
        print(f"  Fork Structure Detection: {'Correct' if enhanced_test['fork_structure']['correct'] else 'Incorrect'}")
        print(f"  V-Structure Detection: {'Correct' if enhanced_test['v_structure']['correct'] else 'Incorrect'}")
        print(f"  Overall Accuracy: {enhanced_test['overall_accuracy']:.2f}")
        
        results = {
            'multi_lag_test': multi_lag_test,
            'enhanced_test': enhanced_test
        }
        
        print("\nTests completed.")
        
    else:
        print("Invalid choice. Please run the script again and select a valid option.")
        return
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    # Show plots if available and if we're not in test mode
    if choice != '4':
        user_input = input("\nWould you like to view the generated plots? (y/n): ")
        if user_input.lower() == 'y':
            plt.show()
    
    print("\nResults have been saved to the 'results' directory.")
    print("\nDone!")
    
    return results

if __name__ == "__main__":
    main() 