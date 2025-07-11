import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

def calculate_expert_activation_ratios_trus(
    expert_indices: torch.Tensor,
    num_experts: int,
    modality_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Calculate the activation ratio for each expert per modality for TRUS model.
    
    Args:
        expert_indices: Tensor of shape (B, M, T, k) containing top-k expert indices
        num_experts: Total number of experts
        modality_names: Optional list of modality names
        
    Returns:
        activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
    """
    B, M, T, k = expert_indices.shape
    
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(M)]
    
    # Initialize activation counts
    activation_counts = np.zeros((num_experts, M))
    
    # Count activations for each expert and modality
    for b in range(B):
        for m in range(M):
            for t in range(T):
                for j in range(k):
                    expert_idx = expert_indices[b, m, t, j].item()
                    activation_counts[expert_idx, m] += 1
    
    # Calculate total tokens per modality
    total_tokens_per_modality = B * T * k
    
    # Calculate activation ratios (percentages)
    activation_ratios = (activation_counts / total_tokens_per_modality) * 100
    
    return activation_ratios


def calculate_expert_activation_ratios_baseline(
    expert_indices: torch.Tensor,
    num_experts: int,
    num_modalities: int,
    seq_len: int,
    modality_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Calculate the activation ratio for each expert per modality for baseline model.
    
    Args:
        expert_indices: Tensor of shape (B, S, k) where S = M * T
        num_experts: Total number of experts
        num_modalities: Number of modalities (M)
        seq_len: Sequence length (T)
        modality_names: Optional list of modality names
        
    Returns:
        activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
    """
    B, S, k = expert_indices.shape
    assert S == num_modalities * seq_len, f"S ({S}) != M ({num_modalities}) * T ({seq_len})"
    
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(num_modalities)]
    
    # Initialize activation counts
    activation_counts = np.zeros((num_experts, num_modalities))
    
    # Count activations for each expert and modality
    # Need to figure out which modality each token belongs to
    for b in range(B):
        for s in range(S):
            # Determine which modality this token belongs to
            # In the model, tokens are reshaped from (B, M, T, E) to (B, M*T, E)
            # So the flattening order is: [M0_T0, M0_T1, ..., M0_T(T-1), M1_T0, M1_T1, ...]
            modality_idx = s // seq_len
            
            for j in range(k):
                expert_idx = expert_indices[b, s, j].item()
                activation_counts[expert_idx, modality_idx] += 1
    
    # Calculate total tokens per modality
    total_tokens_per_modality = B * seq_len * k
    
    # Calculate activation ratios (percentages)
    activation_ratios = (activation_counts / total_tokens_per_modality) * 100
    
    return activation_ratios


def plot_expert_activation_histogram(
    activation_ratios: np.ndarray,
    modality_names: List[str],
    layer_name: str,
    model_type: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot a histogram showing activation ratios for each expert.
    
    Args:
        activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
        modality_names: List of modality names
        layer_name: Name of the MoE layer
        model_type: "TRUS" or "Baseline"
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    num_experts, num_modalities = activation_ratios.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up bar positions
    x = np.arange(num_experts)
    width = 0.8 / num_modalities
    
    # Create color palette
    colors = sns.color_palette("husl", num_modalities)
    
    # Plot bars for each modality
    for i, modality in enumerate(modality_names):
        offset = (i - num_modalities / 2) * width + width / 2
        bars = ax.bar(x + offset, activation_ratios[:, i], width, 
                      label=modality, color=colors[i], alpha=0.8)
        
        # Add value labels on bars if they're significant
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only show labels for bars > 1%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_xlabel('Expert Index', fontsize=24)
    ax.set_ylabel('Activation Ratio (%)', fontsize=24)
    ax.set_title(f'{model_type} Model - {layer_name} Expert Activation Ratios', fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_experts)])
    # ax.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limit
    ax.set_ylim(0, max(100, activation_ratios.max() * 1.1))
    # Set font sizes for ticks
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_all_moe_layers_trus(
    model: nn.Module,
    data_batch: torch.Tensor,
    rus_values: Dict[str, torch.Tensor],
    modality_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None
):
    """
    Plot activation histograms for all MoE layers in a TRUS model.
    
    Args:
        model: The TRUS MoE model
        data_batch: Input data batch of shape (B, M, T, E)
        rus_values: RUS values dictionary
        modality_names: Optional list of modality names
        save_dir: Optional directory to save figures
    """
    model.eval()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # Forward pass to get all auxiliary outputs
        _, all_aux_moe_outputs = model(data_batch, rus_values)
    
    # Get modality names if not provided
    num_modalities = data_batch.shape[1]
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(num_modalities)]
    
    # Plot for each MoE layer
    for layer_idx, aux_outputs in enumerate(all_aux_moe_outputs):
        expert_indices = aux_outputs['expert_indices']
        
        # Get the actual MoE layer to find number of experts
        moe_layer_count = 0
        for layer in model.layers:
            if hasattr(layer, 'moe_layer'):
                if moe_layer_count == layer_idx:
                    num_experts = layer.moe_layer.num_experts
                    break
                moe_layer_count += 1
        
        # Calculate activation ratios
        activation_ratios = calculate_expert_activation_ratios_trus(
            expert_indices, num_experts, modality_names
        )
        
        # Plot histogram
        layer_name = f"MoE Layer {layer_idx + 1}"
        save_path = os.path.join(save_dir, f"trus_moe_layer_{layer_idx + 1}.png") if save_dir else None
        
        plot_expert_activation_histogram(
            activation_ratios,
            modality_names,
            layer_name,
            "TRUS",
            save_path
        )
        
        # Plot stacked activation plot
        stacked_save_path = os.path.join(save_dir, f"trus_moe_layer_{layer_idx + 1}_stacked.png") if save_dir else None
        
        create_stacked_activation_plot(
            activation_ratios,
            modality_names,
            layer_name,
            "TRUS",
            stacked_save_path
        )


def plot_all_moe_layers_baseline(
    model: nn.Module,
    data_batch: torch.Tensor,
    modality_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None
):
    """
    Plot activation histograms for all MoE layers in a baseline model.
    
    Args:
        model: The baseline MoE model
        data_batch: Input data batch of shape (B, M, T, E)
        modality_names: Optional list of modality names
        save_dir: Optional directory to save figures
    """
    model.eval()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    B, M, T, E = data_batch.shape
    
    with torch.no_grad():
        # Forward pass to get all auxiliary outputs
        _, all_aux_moe_outputs = model(data_batch)
    
    # Get modality names if not provided
    if modality_names is None:
        modality_names = [f"Modality_{i}" for i in range(M)]
    
    # Plot for each MoE layer
    for layer_idx, aux_outputs in enumerate(all_aux_moe_outputs):
        expert_indices = aux_outputs['expert_indices']
        
        # Get the actual MoE layer to find number of experts
        moe_layer_count = 0
        for layer in model.layers:
            if hasattr(layer, 'moe_layer'):
                if moe_layer_count == layer_idx:
                    num_experts = layer.moe_layer.num_experts
                    break
                moe_layer_count += 1
        
        # Calculate activation ratios
        activation_ratios = calculate_expert_activation_ratios_baseline(
            expert_indices, num_experts, M, T, modality_names
        )
        
        # Plot histogram
        layer_name = f"MoE Layer {layer_idx + 1}"
        save_path = os.path.join(save_dir, f"baseline_moe_layer_{layer_idx + 1}.png") if save_dir else None
        
        plot_expert_activation_histogram(
            activation_ratios,
            modality_names,
            layer_name,
            "Baseline",
            save_path
        )
        
        # Plot stacked activation plot
        stacked_save_path = os.path.join(save_dir, f"baseline_moe_layer_{layer_idx + 1}_stacked.png") if save_dir else None
        
        create_stacked_activation_plot(
            activation_ratios,
            modality_names,
            layer_name,
            "Baseline",
            stacked_save_path
        )


def create_stacked_activation_plot(
    activation_ratios: np.ndarray,
    modality_names: List[str],
    layer_name: str,
    model_type: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Create a stacked bar plot showing the composition of each expert's activations.
    
    Args:
        activation_ratios: Array of shape (num_experts, num_modalities) with activation percentages
        modality_names: List of modality names
        layer_name: Name of the MoE layer
        model_type: "TRUS" or "Baseline"
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    num_experts, num_modalities = activation_ratios.shape
    
    # Normalize to show composition (percentage of each expert's total activation)
    expert_totals = activation_ratios.sum(axis=1, keepdims=True)
    expert_totals[expert_totals == 0] = 1  # Avoid division by zero
    normalized_ratios = (activation_ratios / expert_totals) * 100
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color palette
    colors = sns.color_palette("husl", num_modalities)
    
    # Create stacked bars
    x = np.arange(num_experts)
    bottom = np.zeros(num_experts)
    
    for i, modality in enumerate(modality_names):
        ax.bar(x, normalized_ratios[:, i], bottom=bottom, 
               label=modality, color=colors[i], alpha=0.8)
        bottom += normalized_ratios[:, i]
    
    # Customize the plot
    ax.set_xlabel('Expert Index', fontsize=24)
    ax.set_ylabel('Modality Composition (%)', fontsize=24)
    ax.set_title(f'{model_type} Model - {layer_name} Expert Modality Composition', fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_experts)])
    # ax.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Example usage function
def analyze_expert_activations(
    trus_model: Optional[nn.Module] = None,
    baseline_model: Optional[nn.Module] = None,
    data_batch: torch.Tensor = None,
    rus_values: Optional[Dict[str, torch.Tensor]] = None,
    modality_names: Optional[List[str]] = None,
    save_dir: str = "../results/expert_activation_plots"
):
    """
    Analyze and plot expert activations for both TRUS and baseline models.
    
    Args:
        trus_model: TRUS MoE model (optional)
        baseline_model: Baseline MoE model (optional)
        data_batch: Input data batch of shape (B, M, T, E)
        rus_values: RUS values dictionary (required for TRUS model)
        modality_names: Optional list of modality names
        save_dir: Directory to save plots
    """
    if data_batch is None:
        print("Error: data_batch is required")
        return
    
    if trus_model is not None:
        if rus_values is None:
            print("Error: rus_values are required for TRUS model")
        else:
            print("Analyzing TRUS model expert activations...")
            trus_save_dir = os.path.join(save_dir, "trus")
            plot_all_moe_layers_trus(trus_model, data_batch, rus_values, 
                                   modality_names, trus_save_dir)
    
    if baseline_model is not None:
        print("Analyzing baseline model expert activations...")
        baseline_save_dir = os.path.join(save_dir, "baseline")
        plot_all_moe_layers_baseline(baseline_model, data_batch, 
                                   modality_names, baseline_save_dir)
    
    print(f"Analysis complete. Plots saved to {save_dir}")
