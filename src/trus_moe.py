import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, Tuple, List, Optional
import numpy as np # Needed for finfo if adapting combine's log/exp space
import pdb

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
    # Calculate KL divergence per element in the batch, then sum over features, then mean over batch
    kl_p_m = F.kl_div(m_log, p, reduction='none', log_target=False).sum(-1)
    kl_q_m = F.kl_div(m_log, q, reduction='none', log_target=False).sum(-1)

    jsd = 0.5 * (kl_p_m + kl_q_m)
    # Average over the batch dimension where JSD was calculated
    return jsd.mean()


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Needs input shape [SeqLen, Batch, EmbedDim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
    """Placeholder for a Synergy Expert."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__(input_dim, hidden_dim, output_dim, dropout_rate)


# --- Router Implementation ---

class RUSAwareGatingNetworkWithAttention(nn.Module):
    """
    A gating network for MoE that incorporates temporal RUS values,
    using attention to aggregate pairwise R and S information.
    Accepts M and T during forward pass to handle flattened sequences.
    """
    def __init__(self, embedding_dim: int, gru_hidden_dim: int, token_processed_dim: int,
                 attn_key_dim: int, attn_value_dim: int, num_experts: int): # Removed num_modalities from init
        super().__init__()
        # Store config but num_modalities comes from input tensor shape
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

    def forward(self, token_embeddings: torch.Tensor, rus_values: Dict[str, torch.Tensor], M: int, T: int) -> torch.Tensor:
        """
        Forward pass of the gating network.
        Args:
            token_embeddings (torch.Tensor): Shape (B, M, T, E) - Expects unflattened here
            rus_values (dict): Contains 'U', 'R', 'S' tensors.
            M (int): Number of modalities.
            T (int): Sequence length per modality.
        Returns:
            torch.Tensor: Routing logits. Shape (B, M, T, N_experts)
        """
        B, _M, _T, E = token_embeddings.shape # Get dimensions
        assert _M == M and _T == T, "Input embedding shape mismatch M/T"
        device = token_embeddings.device
        num_modalities = M # Use passed M

        # Handle edge case M=1 (no pairwise interactions)
        if num_modalities <= 1:
            processed_tokens = self.token_processor(token_embeddings) # (B, 1, T, token_proc_dim)
            rus_temporal_context = torch.zeros(B, num_modalities, T, self.gru_hidden_dim, device=device)
            combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1)
            combined_features_flat = combined_features.view(B * num_modalities * T, -1)
            logits_flat = self.final_mlp(combined_features_flat)
            logits = logits_flat.view(B, num_modalities, T, self.num_experts)
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

        for m_idx in range(num_modalities):
            query_token_repr = processed_tokens[:, m_idx, :, :] # (B, T, token_proc_dim)
            query = self.query_proj(query_token_repr) # (B, T, attn_key_dim)

            other_indices = [j for j in range(num_modalities) if j != m_idx]
            if not other_indices: continue

            R_m_others = R_perm[:, :, m_idx, other_indices] # (B, T, M-1)
            S_m_others = S_perm[:, :, m_idx, other_indices] # (B, T, M-1)
            pairwise_features = torch.stack([R_m_others, S_m_others], dim=-1) # (B, T, M-1, 2)

            pairwise_features_flat = pairwise_features.reshape(-1, 2)
            keys_flat = self.key_proj(pairwise_features_flat)
            values_flat = self.value_proj(pairwise_features_flat)
            keys = keys_flat.view(B, T, num_modalities - 1, self.attn_key_dim) # (B, T, M-1, K_dim)
            values = values_flat.view(B, T, num_modalities - 1, self.attn_value_dim) # (B, T, M-1, V_dim)

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
        gru_input = attn_rus_features.view(B * num_modalities, T, -1)
        rus_temporal_context_flat, _ = self.rus_gru(gru_input)
        rus_temporal_context = rus_temporal_context_flat.view(B, num_modalities, T, self.gru_hidden_dim) # (B, M, T, gru_dim)

        # 6. Combine Token and Temporal RUS Context
        combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1) # (B, M, T, tok_dim + gru_dim)

        # 7. Generate Routing Logits
        combined_features_flat = combined_features.view(B * num_modalities * T, -1)
        logits_flat = self.final_mlp(combined_features_flat)
        logits = logits_flat.view(B, num_modalities, T, self.num_experts) # (B, M, T, N_experts)

        return logits


# --- MoE Layer Implementation ---

class TemporalRUSMoELayer(nn.Module):
    """
    Implements the Mixture-of-Experts layer with RUS-aware gating.
    Uses Top-K routing. Adapts to potentially flattened sequence input.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int, # Output dim of experts must match layer output
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
        self.num_experts = num_experts
        self.num_synergy_experts = num_synergy_experts
        self.k = k

        # Instantiate Experts
        self.experts = nn.ModuleList()
        for _ in range(num_synergy_experts):
            self.experts.append(SynergyExpert(input_dim, expert_hidden_dim, output_dim))
        for _ in range(num_experts - num_synergy_experts):
            self.experts.append(FeedForwardExpert(input_dim, expert_hidden_dim, output_dim))

        self.synergy_expert_indices = set(range(num_synergy_experts))

        # Instantiate Router (remove num_modalities from init)
        self.router = RUSAwareGatingNetworkWithAttention(
            embedding_dim=input_dim, # Router acts on the input embeddings
            num_experts=num_experts,
            # num_modalities=num_modalities, # Removed - determined dynamically
            **router_config
        )

    def forward(self, x: torch.Tensor, rus_values: Dict[str, torch.Tensor], M: int, T: int) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for the MoE layer.

        Args:
            x (torch.Tensor): Input tensor. Shape (B, S, E_in) where S=M*T
            rus_values (Dict[str, torch.Tensor]): Dictionary containing U, R, S tensors.
                                                 Shapes (B,M,T), (B,M,M,T), (B,M,M,T)
            M (int): Original number of modalities.
            T (int): Original sequence length per modality.

        Returns:
            Tuple[torch.Tensor, Dict]:
                - Output tensor. Shape (B, S, E_out)
                - Auxiliary outputs dictionary for loss calculation.
        """
        B, S, E_in = x.shape
        assert S == M * T, f"Sequence length S={S} does not match M*T={M*T}"
        assert E_in == self.input_dim
        device = x.device
        num_tokens = B * S # Total number of tokens in the flattened sequence

        # --- Reshape input for Router ---
        # Router expects (B, M, T, E_in) to correlate with RUS values
        x_unflattened = x.view(B, M, T, E_in)

        # 1. Get routing logits from RUS-aware router
        # Pass M and T explicitly
        router_logits = self.router(x_unflattened, rus_values, M, T) # Shape (B, M, T, N_experts)

        # 2. Perform Top-K Gating
        router_logits_flat = router_logits.view(num_tokens, self.num_experts) # (NumTokens, N_experts)
        gating_probs_flat = F.softmax(router_logits_flat, dim=-1) # (NumTokens, N_experts) - For Aux Losses

        top_k_weights, top_k_indices = torch.topk(router_logits_flat, self.k, dim=-1) # (NumTokens, k), (NumTokens, k)
        top_k_router_probs = F.softmax(top_k_weights, dim=-1) # (NumTokens, k) - Actual routing weights

        # --- 3. Dispatch tokens to experts (Tensor-based approach) ---
        tokens_flat = x.view(num_tokens, E_in) # Use the original flattened input (B*S, E_in)
        final_output_flat = torch.zeros(num_tokens, self.output_dim, device=device)

        flat_expert_indices = top_k_indices.view(-1) # (NumTokens * k)
        flat_router_probs = top_k_router_probs.view(-1) # (NumTokens * k)
        flat_batch_indices = torch.arange(num_tokens, device=device).repeat_interleave(self.k)

        # --- 4. Process inputs for each expert and Combine expert outputs ---
        for expert_idx in range(self.num_experts):
            mask = (flat_expert_indices == expert_idx)
            if mask.any():
                original_token_indices = flat_batch_indices[mask]
                current_routing_probs = flat_router_probs[mask].unsqueeze(-1)
                expert_input = tokens_flat[original_token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_expert_output = expert_output * current_routing_probs
                final_output_flat.index_add_(0, original_token_indices, weighted_expert_output)

        # Reshape final output back to (B, S, E_out)
        final_output = final_output_flat.view(B, S, self.output_dim)

        # Store outputs needed for loss calculation (keep original B,M,T shape for probs)
        aux_outputs = {
            "gating_probs": gating_probs_flat.view(B, M, T, self.num_experts),
            "expert_indices": top_k_indices.view(B, M, T, self.k),
        }

        return final_output, aux_outputs


# --- Standard Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2) # Residual connection
        src = self.norm1(src) # Layer norm

        # Feed Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2) # Residual connection
        src = self.norm2(src) # Layer norm
        return src

# --- TRUS-MoE Transformer Block ---
class TRUSMoEBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, moe_layer: TemporalRUSMoELayer, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_layer = moe_layer # Pass pre-configured MoE layer

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_moe = nn.Dropout(dropout) # Dropout after MoE

    def forward(self, src: torch.Tensor, rus_values: Dict[str, torch.Tensor], M: int, T: int, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Multi-head Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2) # Residual connection
        src = self.norm1(src) # Layer norm

        # TRUS-MoE Layer
        # MoE layer expects (B, S, E_in) and returns (B, S, E_out), plus aux outputs
        # Ensure d_model == moe_layer.input_dim == moe_layer.output_dim
        assert src.size(-1) == self.moe_layer.input_dim, "Dimension mismatch for MoE input"
        moe_output, aux_moe_outputs = self.moe_layer(src, rus_values, M, T)
        assert moe_output.size(-1) == self.moe_layer.output_dim, "Dimension mismatch for MoE output"
        assert src.shape == moe_output.shape, "Shape mismatch after MoE layer"

        src = src + self.dropout_moe(moe_output) # Residual connection
        src = self.norm2(src) # Layer norm

        return src, aux_moe_outputs


# --- Large Scale Model ---
class TRUSMoEModel_LargeScale(nn.Module):
    def __init__(self,
                 input_dim: int, # Dim of input embeddings per token
                 d_model: int, # Internal model dimension
                 nhead: int, # Num heads for MHSA
                 d_ff: int, # Dim for FFN in standard blocks
                 num_encoder_layers: int, # Total number of blocks
                 num_moe_layers: int, # Number of blocks that should use MoE
                 moe_config: Dict, # Config for ALL MoE layers (must match d_model)
                 num_classes: int, # Output classes for classification head
                 dropout: float = 0.1,
                 max_seq_len: int = 1000): # Max combined seq length M*T
        super().__init__()

        assert d_model == moe_config['input_dim'], "MoE input_dim must match d_model"
        assert d_model == moe_config['output_dim'], "MoE output_dim must match d_model"
        self.d_model = d_model

        # Input embedding projection (optional, if input_dim != d_model)
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        self.layers = nn.ModuleList()
        moe_indices = set(random.sample(range(num_encoder_layers), num_moe_layers)) # Randomly choose which layers are MoE

        for i in range(num_encoder_layers):
            if i in moe_indices:
                # Create a new MoE layer instance for each block
                moe_layer_instance = TemporalRUSMoELayer(
                    **moe_config # Unpacks num_experts, num_synergy_experts, k, expert_hidden_dim, router_config
                )
                self.layers.append(TRUSMoEBlock(d_model, nhead, moe_layer_instance, dropout))
            else:
                self.layers.append(TransformerBlock(d_model, nhead, d_ff, dropout))

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)


    def forward(self, token_embeddings: torch.Tensor, rus_values: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass of the large scale model.

        Args:
            token_embeddings (torch.Tensor): Shape (B, M, T, E_in)
            rus_values (Dict[str, torch.Tensor]): Dict with U, R, S tensors.

        Returns:
            Tuple[torch.Tensor, List[Dict]]:
                - Final logits for the task. Shape depends on aggregation (e.g., (B, num_classes)).
                - List of auxiliary outputs from each MoE layer encountered.
        """
        B, M, T, E_in = token_embeddings.shape
        S = M * T # Flattened sequence length

        # 1. Project and Flatten
        x = self.input_proj(token_embeddings) # (B, M, T, d_model)
        x = x * math.sqrt(self.d_model) # Scale embedding
        x = x.view(B, S, self.d_model) # (B, S, d_model)

        # 2. Add Positional Encoding
        # Needs shape (SeqLen, Batch, EmbedDim) for default PositionalEncoding
        x = x.permute(1, 0, 2) # (S, B, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # Back to (B, S, d_model)

        # 3. Pass through stacked layers
        all_aux_moe_outputs = []
        for layer in self.layers:
            if isinstance(layer, TRUSMoEBlock):
                 # Pass M and T explicitly
                x, aux_outputs = layer(x, rus_values, M, T)
                all_aux_moe_outputs.append(aux_outputs)
            elif isinstance(layer, TransformerBlock):
                x = layer(x)
            else:
                raise TypeError("Unsupported layer type in TRUSMoEModel_LargeScale")

        # 4. Final Normalization
        x = self.final_norm(x) # (B, S, d_model)

        # 5. Output Projection & Aggregation (Example: Mean pooling over sequence for classification)
        # Aggregate over the sequence dimension S = M*T
        aggregated_output = x.mean(dim=1) # (B, d_model)
        final_logits = self.output_proj(aggregated_output) # (B, num_classes)

        return final_logits, all_aux_moe_outputs


# --- Loss Calculation Functions (Keep as before) ---

def calculate_rus_losses(gating_probs: torch.Tensor,
                         rus_values: Dict[str, torch.Tensor],
                         synergy_expert_indices: set,
                         threshold_U: float, threshold_R: float, threshold_S: float,
                         lambda_U: float, lambda_R: float, lambda_S: float,
                         epsilon: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates the auxiliary losses based on RUS values and gating probabilities."""
    B, M, T, N_experts = gating_probs.shape
    device = gating_probs.device

    sum_neg_jsd_unique = torch.tensor(0.0, device=device)
    n_unique_pairs = torch.tensor(0.0, device=device)
    sum_jsd_redundant = torch.tensor(0.0, device=device)
    n_redundant_pairs = torch.tensor(0.0, device=device)
    sum_neglogp_synergy = torch.tensor(0.0, device=device)
    n_synergy_pairs = torch.tensor(0.0, device=device)

    U = rus_values['U'] # (B, M, T)
    R = rus_values['R'] # (B, M, M, T)
    S = rus_values['S'] # (B, M, M, T)

    gating_log_probs = F.log_softmax(gating_probs, dim=-1) # (B, M, T, N_exp)
    gating_log_probs_perm = gating_log_probs.permute(0, 2, 1, 3) # (B, T, M, N_exp)

    # --- Uniqueness Loss ---
    if M > 1:
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                indicator = (U[:, m1, :] > threshold_U) & (U[:, m2, :] > threshold_U) # (B, T)
                if indicator.sum() > 0:
                    log_probs_m1 = gating_log_probs_perm[:, :, m1, :][indicator] # (N_valid, N_exp)
                    log_probs_m2 = gating_log_probs_perm[:, :, m2, :][indicator] # (N_valid, N_exp)
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
    """Placeholder for calculating the load balancing loss."""
    if gating_probs.numel() == 0: return torch.tensor(0.0, device=gating_probs.device)
    B, M, T, N_exp = gating_probs.shape
    num_tokens = B * M * T
    probs_flat = gating_probs.view(num_tokens, N_exp)
    f_e = probs_flat.mean(dim=0) # Fraction routed (approx)
    p_e = probs_flat.sum(dim=0) / num_tokens # Mean prob assigned
    load_balance_loss = N_exp * torch.sum(f_e * p_e)
    return lambda_load * load_balance_loss


# --- Startup Example ---

if __name__ == '__main__':
    # --- Configuration ---
    B, M, T, E_in = 2, 3, 10, 64  # Batch, Modalities, SeqLen, InputEmbedDim
    num_classes = 5
    d_model = 128 # Internal dimension
    nhead = 4     # Num heads for MHSA
    d_ff = 256    # FFN dim in standard blocks
    num_encoder_layers = 4 # Total layers
    num_moe_layers = 2     # How many are MoE layers
    dropout = 0.1
    max_seq_len = M * T + 10 # Max sequence length for pos encoding

    # MoE specific config (must match d_model)
    moe_num_experts = 8
    moe_num_synergy_experts = 2
    moe_k = 2
    moe_expert_hidden_dim = 128 # Can differ from d_ff
    moe_router_config = {
        "gru_hidden_dim": 64,
        "token_processed_dim": 64,
        "attn_key_dim": 32,
        "attn_value_dim": 32,
    }
    moe_layer_config = {
        "input_dim": d_model, # Match d_model
        "output_dim": d_model, # Match d_model
        "num_experts": moe_num_experts,
        "num_synergy_experts": moe_num_synergy_experts,
        "k": moe_k,
        "expert_hidden_dim": moe_expert_hidden_dim,
        "router_config": moe_router_config,
    }

    # Loss hyperparameters
    threshold_U, threshold_R, threshold_S = 0.7, 0.7, 0.7
    lambda_U, lambda_R, lambda_S = 0.01, 0.01, 0.01
    lambda_load = 0.01
    epsilon_loss = 1e-8

    # --- Model Instantiation ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRUSMoEModel_LargeScale(
        input_dim=E_in,
        d_model=d_model,
        nhead=nhead,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_moe_layers=num_moe_layers,
        moe_config=moe_layer_config,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    task_criterion = nn.CrossEntropyLoss() # For aggregated output

    # --- Dummy Data ---
    dummy_embeddings = torch.randn(B, M, T, E_in, device=device)
    dummy_rus = {
        'U': torch.rand(B, M, T, device=device),
        'R': torch.rand(B, M, M, T, device=device),
        'S': torch.rand(B, M, M, T, device=device)
    }
    dummy_rus['R'] = 0.5 * (dummy_rus['R'] + dummy_rus['R'].permute(0, 2, 1, 3))
    dummy_rus['S'] = 0.5 * (dummy_rus['S'] + dummy_rus['S'].permute(0, 2, 1, 3))
    # Dummy labels for aggregated classification output
    dummy_labels = torch.randint(0, num_classes, (B,), device=device)

    # --- Training Step Example ---
    model.train()
    optimizer.zero_grad()

    # Forward pass
    final_logits, all_aux_moe_outputs = model(dummy_embeddings, dummy_rus)
    # final_logits shape: (B, num_classes)

    # Calculate Task Loss
    task_loss = task_criterion(final_logits, dummy_labels)

    # Calculate Auxiliary Losses (sum over all MoE layers)
    total_L_unique = torch.tensor(0.0, device=device)
    total_L_redundancy = torch.tensor(0.0, device=device)
    total_L_synergy = torch.tensor(0.0, device=device)
    total_L_load = torch.tensor(0.0, device=device)
    num_moe_layers_encountered = 0

    for i, layer in enumerate(model.layers):
         if isinstance(layer, TRUSMoEBlock):
            # Find the corresponding aux output (assuming order is preserved)
            aux_outputs = all_aux_moe_outputs[num_moe_layers_encountered]
            num_moe_layers_encountered += 1

            # Extract necessary info from this layer's MoE aux output
            gating_probs = aux_outputs['gating_probs']
            expert_indices = aux_outputs['expert_indices']
            synergy_expert_indices = layer.moe_layer.synergy_expert_indices
            k = layer.moe_layer.k

            # Calculate losses for this layer
            L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                gating_probs, dummy_rus, synergy_expert_indices,
                threshold_U, threshold_R, threshold_S,
                lambda_U, lambda_R, lambda_S,
                epsilon=epsilon_loss
            )
            L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, lambda_load)

            total_L_unique += L_unique
            total_L_redundancy += L_redundancy
            total_L_synergy += L_synergy
            total_L_load += L_load

    # Average aux losses over the number of MoE layers
    if num_moe_layers_encountered > 0:
        total_L_unique /= num_moe_layers_encountered
        total_L_redundancy /= num_moe_layers_encountered
        total_L_synergy /= num_moe_layers_encountered
        total_L_load /= num_moe_layers_encountered

    # Combine Losses
    total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load

    # Backward pass and optimize
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("NaN or Inf detected in total loss. Skipping backward pass.")
    else:
        total_loss.backward()
        optimizer.step()

        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"  Task Loss: {task_loss.item():.4f}")
        print(f"  Avg Unique Loss: {total_L_unique.item():.4f}")
        print(f"  Avg Redundancy Loss: {total_L_redundancy.item():.4f}")
        print(f"  Avg Synergy Loss: {total_L_synergy.item():.4f}")
        print(f"  Avg Load Balancing Loss: {total_L_load.item():.4f}")
