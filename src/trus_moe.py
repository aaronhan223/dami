import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, Tuple, List, Optional
import numpy as np # Needed for finfo if adapting combine's log/exp space

# --- Helper Functions ---

def JSD(p_log: torch.Tensor, q_log: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculates Jensen-Shannon Divergence between two log-probability distributions.
    Uses the formula: JSD(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)] where M = 0.5*(P+Q)
    Expects log-probabilities as input for stability with F.kl_div.
    """
    p = torch.exp(p_log)
    q = torch.exp(q_log)
    m = 0.5 * (p + q)
    # Clamp m to avoid log(0) issues
    m_log = torch.log(m.clamp(min=eps))

    # F.kl_div expects input=log_probs, target=probs
    kl_p_m = F.kl_div(m_log, p, reduction='none', log_target=False).sum(-1)
    kl_q_m = F.kl_div(m_log, q, reduction='none', log_target=False).sum(-1)

    jsd = 0.5 * (kl_p_m + kl_q_m)
    # Average over the batch dimension where JSD was calculated
    return jsd.mean()


# --- Expert Implementations ---

class FeedForwardExpert(nn.Module):
    """A simple Feed-Forward Network expert."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SynergyExpert(FeedForwardExpert):
    """
    Placeholder for a Synergy Expert.
    In this example, it behaves like a standard FFN.
    A real implementation might take multiple inputs or use cross-attention.
    We designate it mainly so the L_synergy loss can target it.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__(input_dim, hidden_dim, output_dim, dropout_rate)
        # Add specific parameters or layers for synergy here if needed
        pass

# --- Router Implementation ---

class RUSAwareGatingNetworkWithAttention(nn.Module):
    """
    A gating network for MoE that incorporates temporal RUS values,
    using attention to aggregate pairwise R and S information.
    """
    def __init__(self, embedding_dim: int, gru_hidden_dim: int, token_processed_dim: int,
                 attn_key_dim: int, attn_value_dim: int, num_experts: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_experts = num_experts
        self.attn_key_dim = attn_key_dim
        self.attn_value_dim = attn_value_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.token_processed_dim = token_processed_dim

        self.token_processor = nn.Sequential(
            nn.Linear(embedding_dim, token_processed_dim),
            nn.ReLU()
        )
        self.query_proj = nn.Linear(token_processed_dim, attn_key_dim)
        self.key_proj = nn.Linear(2, attn_key_dim) # Input is (R, S) pair
        self.value_proj = nn.Linear(2, attn_value_dim)
        self.rus_gru = nn.GRU(
            input_size=1 + attn_value_dim, # U + AttnContext(R,S)
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        combined_mlp_input_dim = token_processed_dim + gru_hidden_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_mlp_input_dim, combined_mlp_input_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_mlp_input_dim // 2, num_experts)
        )

    def forward(self, token_embeddings: torch.Tensor, rus_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the gating network.
        Args:
            token_embeddings (torch.Tensor): Shape (B, M, T, E)
            rus_values (dict): Contains 'U', 'R', 'S' tensors.
        Returns:
            torch.Tensor: Routing logits. Shape (B, M, T, N_experts)
        """
        B, M, T, E = token_embeddings.shape
        device = token_embeddings.device

        # Handle edge case M=1 (no pairwise interactions)
        if M <= 1:
            processed_tokens = self.token_processor(token_embeddings) # (B, 1, T, token_proc_dim)
            # Create zero context matching GRU output dim
            rus_temporal_context = torch.zeros(B, M, T, self.gru_hidden_dim, device=device)
            combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1)
            combined_features_flat = combined_features.view(B * M * T, -1)
            logits_flat = self.final_mlp(combined_features_flat)
            logits = logits_flat.view(B, M, T, self.num_experts)
            return logits

        U = rus_values['U'] # (B, M, T)
        R = rus_values['R'] # (B, M, M, T)
        S = rus_values['S'] # (B, M, M, T)

        # 1. Prepare Pairwise RUS Features (handled inside loop)
        # 2. Process Token Embeddings
        processed_tokens = self.token_processor(token_embeddings) # (B, M, T, token_proc_dim)

        # 3. Compute Attention Context for R/S
        all_attn_contexts = []
        R_perm = R.permute(0, 3, 1, 2) # (B, T, M, M)
        S_perm = S.permute(0, 3, 1, 2) # (B, T, M, M)

        for m_idx in range(M):
            query_token_repr = processed_tokens[:, m_idx, :, :] # (B, T, token_proc_dim)
            query = self.query_proj(query_token_repr) # (B, T, attn_key_dim)

            other_indices = [j for j in range(M) if j != m_idx]
            if not other_indices: continue # Should not happen if M > 1

            R_m_others = R_perm[:, :, m_idx, other_indices] # (B, T, M-1)
            S_m_others = S_perm[:, :, m_idx, other_indices] # (B, T, M-1)
            pairwise_features = torch.stack([R_m_others, S_m_others], dim=-1) # (B, T, M-1, 2)

            pairwise_features_flat = pairwise_features.reshape(-1, 2)
            keys_flat = self.key_proj(pairwise_features_flat)
            values_flat = self.value_proj(pairwise_features_flat)
            keys = keys_flat.view(B, T, M - 1, self.attn_key_dim) # (B, T, M-1, K_dim)
            values = values_flat.view(B, T, M - 1, self.attn_value_dim) # (B, T, M-1, V_dim)

            query_unsqueezed = query.unsqueeze(2) # (B, T, 1, Q_dim)
            attn_scores = torch.matmul(query_unsqueezed, keys.transpose(-2, -1)) / math.sqrt(self.attn_key_dim) # (B, T, 1, M-1)
            attn_weights = F.softmax(attn_scores, dim=-1) # (B, T, 1, M-1)
            attn_context = torch.matmul(attn_weights, values).squeeze(2) # (B, T, V_dim)
            all_attn_contexts.append(attn_context)

        stacked_attn_contexts = torch.stack(all_attn_contexts, dim=1) # (B, M, T, V_dim)

        # 4. Create Attn-RUS Feature Sequence
        U_unsqueezed = U.unsqueeze(-1) # (B, M, T, 1)
        attn_rus_features = torch.cat([U_unsqueezed, stacked_attn_contexts], dim=-1) # (B, M, T, 1 + V_dim)

        # 5. Process Temporal Dynamics
        gru_input = attn_rus_features.view(B * M, T, -1)
        rus_temporal_context_flat, _ = self.rus_gru(gru_input)
        rus_temporal_context = rus_temporal_context_flat.view(B, M, T, self.gru_hidden_dim) # (B, M, T, gru_dim)

        # 6. Combine Token and Temporal RUS Context
        combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1) # (B, M, T, tok_dim + gru_dim)

        # 7. Generate Routing Logits
        combined_features_flat = combined_features.view(B * M * T, -1)
        logits_flat = self.final_mlp(combined_features_flat)
        logits = logits_flat.view(B, M, T, self.num_experts) # (B, M, T, N_experts)

        return logits


# --- MoE Layer Implementation ---

class TemporalRUSMoELayer(nn.Module):
    """
    Implements the Mixture-of-Experts layer with RUS-aware gating.
    Uses Top-K routing with updated dispatch/combine logic.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int, # Output dim of experts must match layer output
                 num_modalities: int,
                 num_experts: int,
                 num_synergy_experts: int, # How many of the experts are designated synergy experts
                 k: int, # Number of experts to route each token to
                 expert_hidden_dim: int,
                 router_config: Dict): # Config for RUSAwareGatingNetworkWithAttention
        super().__init__()
        assert num_experts > 0
        assert 0 < k <= num_experts
        assert 0 <= num_synergy_experts <= num_experts

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities
        self.num_experts = num_experts
        self.num_synergy_experts = num_synergy_experts
        self.k = k

        # Instantiate Experts
        self.experts = nn.ModuleList()
        # Synergy experts first (if any)
        for _ in range(num_synergy_experts):
             # Using placeholder SynergyExpert for now
            self.experts.append(SynergyExpert(input_dim, expert_hidden_dim, output_dim))
        # Standard experts
        for _ in range(num_experts - num_synergy_experts):
            self.experts.append(FeedForwardExpert(input_dim, expert_hidden_dim, output_dim))

        # Identify indices of synergy experts
        self.synergy_expert_indices = set(range(num_synergy_experts))

        # Instantiate Router
        self.router = RUSAwareGatingNetworkWithAttention(
            embedding_dim=input_dim, # Router acts on the input embeddings
            num_experts=num_experts,
            num_modalities=num_modalities,
            **router_config # Pass other params like gru_hidden_dim etc.
        )

    def forward(self, token_embeddings: torch.Tensor, rus_values: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for the MoE layer with updated dispatch/combine.

        Args:
            token_embeddings (torch.Tensor): Input tensor. Shape (B, M, T, E_in)
            rus_values (Dict[str, torch.Tensor]): Dictionary containing U, R, S tensors.

        Returns:
            Tuple[torch.Tensor, Dict]:
                - Output tensor. Shape (B, M, T, E_out)
                - Auxiliary outputs dictionary containing 'gating_probs', 'expert_indices', etc. for loss calculation.
        """
        B, M, T, E_in = token_embeddings.shape
        assert E_in == self.input_dim
        device = token_embeddings.device
        num_tokens = B * M * T

        # 1. Get routing logits from RUS-aware router
        router_logits = self.router(token_embeddings, rus_values) # Shape (B, M, T, N_experts)

        # 2. Perform Top-K Gating
        router_logits_flat = router_logits.view(num_tokens, self.num_experts)
        gating_probs_flat = F.softmax(router_logits_flat, dim=-1) # (NumTokens, N_experts) - For Aux Losses

        top_k_weights, top_k_indices = torch.topk(router_logits_flat, self.k, dim=-1) # (NumTokens, k), (NumTokens, k)
        top_k_router_probs = F.softmax(top_k_weights, dim=-1) # (NumTokens, k) - Actual routing weights

        # --- 3. Dispatch tokens to experts (Tensor-based approach) ---
        tokens_flat = token_embeddings.view(num_tokens, E_in) # (NumTokens, E_in)
        final_output_flat = torch.zeros(num_tokens, self.output_dim, device=device)

        # Flatten indices and probs corresponding to the K selected experts for each token
        flat_expert_indices = top_k_indices.view(-1) # (NumTokens * k)
        flat_router_probs = top_k_router_probs.view(-1) # (NumTokens * k)
        # Create batch index mapping: [0, 0, ..., 1, 1, ..., NumTokens-1, ...] length NumTokens*k
        flat_batch_indices = torch.arange(num_tokens, device=device).repeat_interleave(self.k)

        # --- 4. Process inputs for each expert and Combine expert outputs ---
        for expert_idx in range(self.num_experts):
            # Find which flattened assignments correspond to the current expert
            mask = (flat_expert_indices == expert_idx)

            if mask.any(): # Check if any tokens are routed to this expert
                # Get the original token indices for this expert's batch
                original_token_indices = flat_batch_indices[mask] # Indices into tokens_flat

                # Get the routing probabilities for these tokens for this expert
                current_routing_probs = flat_router_probs[mask].unsqueeze(-1) # (N_for_expert, 1)

                # Gather the input tokens for this expert
                expert_input = tokens_flat[original_token_indices] # (N_for_expert, E_in)

                # Pass tokens through the expert
                expert_output = self.experts[expert_idx](expert_input) # (N_for_expert, E_out)

                # Weight the expert output by the routing probability
                weighted_expert_output = expert_output * current_routing_probs # (N_for_expert, E_out)

                # Add (scatter) the weighted outputs back to the final output tensor
                # using the original token indices. index_add_ handles summation for k > 1.
                final_output_flat.index_add_(0, original_token_indices, weighted_expert_output)

        # Reshape final output back to (B, M, T, E_out)
        final_output = final_output_flat.view(B, M, T, self.output_dim)

        # Store outputs needed for loss calculation
        aux_outputs = {
            # Return full probabilities for loss calculation
            "gating_probs": gating_probs_flat.view(B, M, T, self.num_experts),
            "expert_indices": top_k_indices.view(B, M, T, self.k),
        }

        return final_output, aux_outputs


# --- Loss Calculation Functions ---

def calculate_rus_losses(gating_probs: torch.Tensor,
                         rus_values: Dict[str, torch.Tensor],
                         synergy_expert_indices: set,
                         threshold_U: float, threshold_R: float, threshold_S: float,
                         lambda_U: float, lambda_R: float, lambda_S: float,
                         epsilon: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates the auxiliary losses based on RUS values and gating probabilities."""
    B, M, T, N_experts = gating_probs.shape
    device = gating_probs.device

    # Ensure calculations happen on the correct device
    sum_neg_jsd_unique = torch.tensor(0.0, device=device)
    n_unique_pairs = torch.tensor(0.0, device=device)
    sum_jsd_redundant = torch.tensor(0.0, device=device)
    n_redundant_pairs = torch.tensor(0.0, device=device)
    sum_neglogp_synergy = torch.tensor(0.0, device=device)
    n_synergy_pairs = torch.tensor(0.0, device=device)

    U = rus_values['U'] # (B, M, T)
    R = rus_values['R'] # (B, M, M, T)
    S = rus_values['S'] # (B, M, M, T)

    # Use log_softmax for numerical stability in JSD calculation
    gating_log_probs = F.log_softmax(gating_probs, dim=-1) # (B, M, T, N_exp)
    gating_log_probs_perm = gating_log_probs.permute(0, 2, 1, 3) # (B, T, M, N_exp)

    # --- Uniqueness Loss ---
    if M > 1:
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                indicator = (U[:, m1, :] > threshold_U) & (U[:, m2, :] > threshold_U) # (B, T)
                if indicator.sum() > 0:
                    # Select log probabilities for valid pairs
                    log_probs_m1 = gating_log_probs_perm[:, :, m1, :][indicator] # (N_valid, N_exp)
                    log_probs_m2 = gating_log_probs_perm[:, :, m2, :][indicator] # (N_valid, N_exp)
                    # Calculate JSD (expects log probs)
                    jsd_values = JSD(log_probs_m1, log_probs_m2) # Returns scalar mean over N_valid
                    sum_neg_jsd_unique += (-jsd_values) * indicator.sum() # Scale mean by count
                    n_unique_pairs += indicator.sum()

    L_unique = lambda_U * (sum_neg_jsd_unique / (n_unique_pairs + epsilon))

    # --- Redundancy Loss ---
    if M > 1:
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                indicator = (R[:, m1, m2, :] > threshold_R) # (B, T)
                if indicator.sum() > 0:
                    log_probs_m1 = gating_log_probs_perm[:, :, m1, :][indicator]
                    log_probs_m2 = gating_log_probs_perm[:, :, m2, :][indicator]
                    jsd_values = JSD(log_probs_m1, log_probs_m2)
                    sum_jsd_redundant += jsd_values * indicator.sum()
                    n_redundant_pairs += indicator.sum()

    L_redundancy = lambda_R * (sum_jsd_redundant / (n_redundant_pairs + epsilon))

    # --- Synergy Loss ---
    synergy_expert_indices_list = list(synergy_expert_indices)
    if M > 1 and synergy_expert_indices_list:
        # Use original probabilities (not log) for summing
        gating_probs_perm = gating_probs.permute(0, 2, 1, 3) # (B, T, M, N_exp)
        synergy_probs = gating_probs_perm[:, :, :, synergy_expert_indices_list] # (B, T, M, N_syn_exp)
        p_assign_synergy_all = torch.sum(synergy_probs, dim=-1) # (B, T, M)

        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                indicator = (S[:, m1, m2, :] > threshold_S) # (B, T)
                if indicator.sum() > 0:
                    p_syn_m1 = p_assign_synergy_all[:, :, m1][indicator] # (N_valid,)
                    p_syn_m2 = p_assign_synergy_all[:, :, m2][indicator] # (N_valid,)
                    avg_p_synergy = (p_syn_m1 + p_syn_m2) / 2.0
                    neg_log_p = -torch.log(avg_p_synergy.clamp(min=epsilon)) # Clamp before log
                    sum_neglogp_synergy += neg_log_p.sum()
                    n_synergy_pairs += indicator.sum()

    L_synergy = lambda_S * (sum_neglogp_synergy / (n_synergy_pairs + epsilon))

    return L_unique, L_redundancy, L_synergy


def calculate_load_balancing_loss(gating_probs: torch.Tensor, expert_indices: torch.Tensor, k: int, lambda_load: float) -> torch.Tensor:
    """
    Placeholder for calculating the load balancing loss.
    Args:
        gating_probs (torch.Tensor): Softmax probabilities from router (B, M, T, N_exp)
        expert_indices (torch.Tensor): Indices of top-k selected experts (B, M, T, k)
        k (int): Top-k value
        lambda_load (float): Weight for the loss.
    Returns:
        torch.Tensor: Load balancing loss value.
    """
    if gating_probs.numel() == 0: return torch.tensor(0.0, device=gating_probs.device)
    B, M, T, N_exp = gating_probs.shape
    num_tokens = B * M * T

    probs_flat = gating_probs.view(num_tokens, N_exp)
    f_e = probs_flat.mean(dim=0) # Fraction routed (approx)
    p_e = probs_flat.sum(dim=0) / num_tokens # Mean prob assigned

    # Simple load balance loss (Switch MoE paper inspired)
    load_balance_loss = N_exp * torch.sum(f_e * p_e)

    return lambda_load * load_balance_loss


# --- Overall Model Example ---

class TRUSMoEModel(nn.Module):
    """Example model incorporating the Temporal RUS MoE Layer."""
    def __init__(self, input_dim: int, num_modalities: int, num_classes: int, moe_config: Dict):
        super().__init__()
        self.num_modalities = num_modalities
        self.input_proj = nn.Linear(input_dim, moe_config['input_dim'])
        self.moe_layer = TemporalRUSMoELayer(num_modalities=num_modalities, **moe_config)
        self.output_proj = nn.Linear(moe_config['output_dim'], num_classes)

    def forward(self, token_embeddings: torch.Tensor, rus_values: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        B, M, T, E_in = token_embeddings.shape
        projected_embeddings = self.input_proj(token_embeddings)
        moe_output, aux_moe_outputs = self.moe_layer(projected_embeddings, rus_values)
        moe_output_flat = moe_output.view(-1, moe_output.size(-1))
        final_logits_flat = self.output_proj(moe_output_flat)
        final_logits = final_logits_flat.view(B, M, T, -1)
        return final_logits, aux_moe_outputs

# --- Startup Example ---

if __name__ == '__main__':
    # --- Configuration ---
    B, M, T, E_in = 2, 3, 5, 32  # Batch, Modalities, SeqLen, InputEmbedDim
    num_classes = 10
    num_experts = 4
    num_synergy_experts = 1 # First expert is synergy expert
    k = 2 # Top-k routing
    expert_hidden_dim = 64
    moe_input_dim = 32 # Dimension expected by MoE layer experts
    moe_output_dim = 32 # Dimension output by MoE layer experts

    router_config = {
        "gru_hidden_dim": 24,
        "token_processed_dim": 32,
        "attn_key_dim": 16,
        "attn_value_dim": 16,
    }

    moe_config = {
        "input_dim": moe_input_dim,
        "output_dim": moe_output_dim,
        "num_experts": num_experts,
        "num_synergy_experts": num_synergy_experts,
        "k": k,
        "expert_hidden_dim": expert_hidden_dim,
        "router_config": router_config,
    }

    # Loss hyperparameters
    threshold_U, threshold_R, threshold_S = 0.7, 0.7, 0.7
    lambda_U, lambda_R, lambda_S = 0.01, 0.01, 0.01
    lambda_load = 0.01 # Weight for load balancing loss
    epsilon_loss = 1e-8 # Epsilon for loss calculations

    # --- Model Instantiation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRUSMoEModel(E_in, M, num_classes, moe_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    task_criterion = nn.CrossEntropyLoss()

    # --- Dummy Data ---
    dummy_embeddings = torch.randn(B, M, T, E_in, device=device)
    dummy_rus = {
        'U': torch.rand(B, M, T, device=device),
        'R': torch.rand(B, M, M, T, device=device),
        'S': torch.rand(B, M, M, T, device=device)
    }
    dummy_rus['R'] = 0.5 * (dummy_rus['R'] + dummy_rus['R'].permute(0, 2, 1, 3))
    dummy_rus['S'] = 0.5 * (dummy_rus['S'] + dummy_rus['S'].permute(0, 2, 1, 3))
    dummy_labels = torch.randint(0, num_classes, (B * M * T,), device=device)

    # --- Training Step Example ---
    model.train()
    optimizer.zero_grad()

    # Forward pass
    final_logits, aux_moe_outputs = model(dummy_embeddings, dummy_rus)

    # Calculate Task Loss
    task_loss = task_criterion(final_logits.view(-1, num_classes), dummy_labels)

    # Calculate Auxiliary RUS Losses
    synergy_expert_indices = model.moe_layer.synergy_expert_indices
    gating_probs = aux_moe_outputs['gating_probs'] # Shape (B, M, T, N_experts)

    L_unique, L_redundancy, L_synergy = calculate_rus_losses(
        gating_probs, dummy_rus, synergy_expert_indices,
        threshold_U, threshold_R, threshold_S,
        lambda_U, lambda_R, lambda_S,
        epsilon=epsilon_loss
    )

    # Calculate Load Balancing Loss
    expert_indices = aux_moe_outputs['expert_indices'] # Shape (B, M, T, k)
    L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, lambda_load)

    # Combine Losses
    total_loss = task_loss + L_unique + L_redundancy + L_synergy + L_load

    # Backward pass and optimize
    # Check for NaNs/Infs before backward pass
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("NaN or Inf detected in total loss. Skipping backward pass.")
        # Optionally print individual losses to debug
        print(f"  Task Loss: {task_loss.item()}")
        print(f"  Unique Loss: {L_unique.item()}")
        print(f"  Redundancy Loss: {L_redundancy.item()}")
        print(f"  Synergy Loss: {L_synergy.item()}")
        print(f"  Load Balancing Loss: {L_load.item()}")
    else:
        total_loss.backward()
        optimizer.step()

        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"  Task Loss: {task_loss.item():.4f}")
        print(f"  Unique Loss: {L_unique.item():.4f}")
        print(f"  Redundancy Loss: {L_redundancy.item():.4f}")
        print(f"  Synergy Loss: {L_synergy.item():.4f}")
        print(f"  Load Balancing Loss: {L_load.item():.4f}")
