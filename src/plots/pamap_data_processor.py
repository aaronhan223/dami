"""
PAMAP Data Processor for RUS Plotting

This script contains functions to process PAMAP RUS data so it can be used
with the plot_multi_lag_results function from analyze_mimiciv_rus.ipynb.

The main difference is that PAMAP data has a nested structure where each
modality pair contains a 'lag_results' list, while MIMIC-IV data has a
flat structure where each entry represents one lag result.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_pamap_rus_data(file_path):
    """
    Load PAMAP RUS data from numpy file.

    Args:
        file_path (str): Path to the PAMAP RUS numpy file

    Returns:
        numpy.ndarray: Loaded PAMAP RUS data
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Successfully loaded {len(data)} modality pairs from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def find_chest_hand_pair(pamap_data):
    """
    Find the chest vs hand modality pair in PAMAP data.

    Args:
        pamap_data (numpy.ndarray): PAMAP RUS data

    Returns:
        int: Index of chest vs hand pair, or -1 if not found
    """
    if pamap_data is None:
        return -1

    for i, pair_data in enumerate(pamap_data):
        feature_pair = pair_data['feature_pair']
        if feature_pair == ('chest', 'hand'):
            return i

    print("Chest vs hand pair not found. Available pairs:")
    for i, pair_data in enumerate(pamap_data):
        feature_pair = pair_data['feature_pair']
        print(f"  {i}: {feature_pair[0]} → {feature_pair[1]}")

    return -1


def process_pamap_chest_hand_for_plotting(pamap_data):
    """
    Process PAMAP chest vs hand data to be compatible with plot_multi_lag_results function.

    Args:
        pamap_data (numpy.ndarray): PAMAP RUS data

    Returns:
        list: List of dictionaries in MIMIC-IV format for plotting
    """
    if pamap_data is None:
        print("No PAMAP data available")
        return []

    # Find chest vs hand pair
    pair_index = find_chest_hand_pair(pamap_data)
    if pair_index == -1:
        return []

    # Get the chest vs hand pair data
    pair_data = pamap_data[pair_index]

    # Extract information
    feature_pair = pair_data['feature_pair']
    lag_results = pair_data['lag_results']

    print(f"Processing {feature_pair[0]} → {feature_pair[1]} pair")
    print(f"Number of features: {pair_data['n_features_mod1']} vs {pair_data['n_features_mod2']}")
    print(f"Number of lags: {len(lag_results)}")

    # Convert to MIMIC-IV format for plotting
    processed_data = []
    for lag_result in lag_results:
        # Determine dominant term (component with highest value)
        values = {
            'R': lag_result['R_value'],
            'U1': lag_result['U1_value'],
            'U2': lag_result['U2_value'],
            'S': lag_result['S_value']
        }
        dominant_term = max(values, key=values.get)

        # Create entry in MIMIC-IV format
        entry = {
            'feature_pair': feature_pair,
            'lag': lag_result['lag'],
            'dominant_term': dominant_term,
            'R_value': lag_result['R_value'],
            'U1_value': lag_result['U1_value'],
            'U2_value': lag_result['U2_value'],
            'S_value': lag_result['S_value'],
            'MI_value': lag_result['MI_value']
        }
        processed_data.append(entry)

    return processed_data


def plot_multi_lag_results(feature_pair_data, feature_pair_name, ax=None):
    """
    Plot RUS decomposition results for different lags of a feature pair.
    (Copied from analyze_mimiciv_rus.ipynb)

    Args:
        feature_pair_data: Array of dictionaries containing RUS results for different lags
        feature_pair_name: String name of the feature pair for the title
        ax: Matplotlib axis object (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data for plotting
    lags = [item['lag'] for item in feature_pair_data]
    r_values = [item['R_value'] for item in feature_pair_data]
    u1_values = [item['U1_value'] for item in feature_pair_data]
    u2_values = [item['U2_value'] for item in feature_pair_data]
    s_values = [item['S_value'] for item in feature_pair_data]
    mi_values = [item['MI_value'] for item in feature_pair_data]

    # Create the line plot
    ax.plot(lags, r_values, 'o-', label='R (Redundancy)', linewidth=4, markersize=12)
    ax.plot(lags, u1_values, 'o-', label='U₁ (Unique X)', linewidth=4, markersize=12)
    ax.plot(lags, u2_values, 'o-', label='U₂ (Unique Y)', linewidth=4, markersize=12)
    ax.plot(lags, s_values, 'o-', label='S (Synergy)', linewidth=4, markersize=12)
    ax.plot(lags, mi_values, 'o-', label='MI (Total)', linewidth=4, color='black', linestyle='--', markersize=12)

    # Annotate the dominant term
    for i, item in enumerate(feature_pair_data):
        dom_term = item['dominant_term']
        # Position the annotation slightly higher above the MI value
        ax.annotate(dom_term, (lags[i], mi_values[i] + 0.02 * max(mi_values)),
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', fontweight='bold', fontsize=20)

    # Add labels and title
    ax.set_xlabel('Time Lag', fontsize=22)
    ax.set_ylabel('Information (bits)', fontsize=22)
    ax.set_title(f"{feature_pair_name}", fontsize=25)

    # Add legend and grid
    ax.legend(loc='upper right', fontsize=22)
    ax.grid(True, linestyle='--', alpha=0.7)
    # Set x and y ticks font size
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    return ax


def main():
    """
    Main function to demonstrate the PAMAP data processing and plotting.
    """
    # File path
    pamap_data_file = "/cis/home/xhan56/code/dami/results/pamap/pamap_subject3_multimodal_all_lag10_bins4.npy"

    # Load PAMAP data
    print("Loading PAMAP RUS data...")
    pamap_data = load_pamap_rus_data(pamap_data_file)

    if pamap_data is None:
        return

    # Process chest vs hand data for plotting
    print("\nProcessing chest vs hand data for plotting...")
    processed_data = process_pamap_chest_hand_for_plotting(pamap_data)

    if not processed_data:
        print("Failed to process data")
        return

    # Create plot
    print("\nCreating plot...")
    pair_name = f"{processed_data[0]['feature_pair'][0]} → {processed_data[0]['feature_pair'][1]}"
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_multi_lag_results(processed_data, pair_name, ax)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSummary for {pair_name}:")
    print(f"Lags analyzed: {len(processed_data)}")
    print("Dominant terms by lag:")
    for item in processed_data:
        print(f"  Lag {item['lag']}: {item['dominant_term']} (MI = {item['MI_value']:.4f} bits)")


if __name__ == "__main__":
    main()