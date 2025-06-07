import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
from tqdm import tqdm


def JSD(p_log: torch.Tensor, q_log: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculates Jensen-Shannon Divergence between two log-probability distributions.
    Uses the formula: JSD(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)] where M = 0.5*(P+Q)
    Expects log-probabilities as input for stability with F.kl_div.
    """
    # Ensure inputs are log-probabilities
    # p_log = F.log_softmax(p_log, dim=-1)
    # q_log = F.log_softmax(q_log, dim=-1)
    p = torch.exp(p_log)
    q = torch.exp(q_log)
    m = 0.5 * (p + q)
    m_log = torch.log(m.clamp(min=eps)) # Clamp m before log

    # F.kl_div expects input=log_probs, target=probs
    kl_p_m = F.kl_div(m_log, p, reduction='none', log_target=False).sum(-1)
    kl_q_m = F.kl_div(m_log, q, reduction='none', log_target=False).sum(-1)

    jsd = 0.5 * (kl_p_m + kl_q_m)
    # Average over the batch dimension where JSD was calculated
    # Handle potential empty batch (if indicator selects nothing)
    if jsd.numel() == 0:
        return torch.tensor(0.0, device=jsd.device)
    return jsd.mean()


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

class SynergyExpert(nn.Module):
    """
    An expert designed to handle synergistic interactions using self-attention.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nhead: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        assert input_dim == output_dim, "SynergyExpert currently requires input_dim == output_dim for residual connection"
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=nhead, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N_for_expert = x.size(0)
        if N_for_expert <= 1:
            attn_output = x
        else:
            # Treat N_for_expert as sequence length for MHA (batch size 1)
            x_attn_input = x.unsqueeze(0) # Shape (1, N_for_expert, E_in)
            attn_output, _ = self.self_attn(x_attn_input, x_attn_input, x_attn_input, need_weights=False)
            attn_output = attn_output.squeeze(0) # Back to (N_for_expert, E_in)

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


# --- Router Implementation ---
class RUSAwareGatingNetworkWithAttention(nn.Module):
    """
    RUS-aware gating network using attention for pairwise R/S aggregation.
    """
    def __init__(self, embedding_dim: int, gru_hidden_dim: int, token_processed_dim: int,
                 attn_key_dim: int, attn_value_dim: int, num_experts: int):
        super().__init__()
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
        self.key_proj = nn.Linear(2, attn_key_dim)
        self.value_proj = nn.Linear(2, attn_value_dim)
        self.rus_gru = nn.GRU(
            input_size=1 + attn_value_dim,
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
        B, _M, _T, E = token_embeddings.shape
        assert _M == M and _T == T, "Input embedding shape mismatch M/T"
        device = token_embeddings.device
        num_modalities = M

        if num_modalities <= 1:
            processed_tokens = self.token_processor(token_embeddings)
            rus_temporal_context = torch.zeros(B, num_modalities, T, self.gru_hidden_dim, device=device)
            combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1)
            logits = self.final_mlp(combined_features.view(B * M * T, -1))
            return logits.view(B, M, T, self.num_experts)

        U = rus_values['U']
        R = rus_values['R']
        S = rus_values['S']
        processed_tokens = self.token_processor(token_embeddings)

        all_attn_contexts = []
        R_perm = R.permute(0, 3, 1, 2)
        S_perm = S.permute(0, 3, 1, 2)

        for m_idx in range(num_modalities):
            query_token_repr = processed_tokens[:, m_idx, :, :]
            query = self.query_proj(query_token_repr)
            other_indices = [j for j in range(num_modalities) if j != m_idx]
            if not other_indices: continue

            R_m_others = R_perm[:, :, m_idx, other_indices]
            S_m_others = S_perm[:, :, m_idx, other_indices]
            pairwise_features = torch.stack([R_m_others, S_m_others], dim=-1)

            keys_flat = self.key_proj(pairwise_features.view(-1, 2))
            values_flat = self.value_proj(pairwise_features.view(-1, 2))
            keys = keys_flat.view(B, T, num_modalities - 1, self.attn_key_dim)
            values = values_flat.view(B, T, num_modalities - 1, self.attn_value_dim)

            query_unsqueezed = query.unsqueeze(2)
            attn_scores = torch.matmul(query_unsqueezed, keys.transpose(-2, -1)) / math.sqrt(self.attn_key_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_context = torch.matmul(attn_weights, values).squeeze(2)
            all_attn_contexts.append(attn_context)

        stacked_attn_contexts = torch.stack(all_attn_contexts, dim=1)
        U_unsqueezed = U.unsqueeze(-1)
        attn_rus_features = torch.cat([U_unsqueezed, stacked_attn_contexts], dim=-1)

        gru_input = attn_rus_features.view(B * num_modalities, T, -1)
        rus_temporal_context_flat, _ = self.rus_gru(gru_input)
        rus_temporal_context = rus_temporal_context_flat.view(B, num_modalities, T, self.gru_hidden_dim)

        combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1)
        logits = self.final_mlp(combined_features.view(B * M * T, -1))
        return logits.view(B, M, T, self.num_experts)


# --- MoE Layer Implementation ---
class TemporalRUSMoELayer(nn.Module):
    """
    Implements the Mixture-of-Experts layer with RUS-aware gating.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_experts: int,
                 num_synergy_experts: int,
                 k: int,
                 expert_hidden_dim: int,
                 synergy_expert_nhead: int,
                 router_config: Dict):
        super().__init__()
        assert num_experts > 0 and 0 < k <= num_experts and 0 <= num_synergy_experts <= num_experts
        assert input_dim == output_dim, "MoE Layer requires input_dim == output_dim for residual connections"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList()
        for _ in range(num_synergy_experts):
            self.experts.append(SynergyExpert(input_dim, expert_hidden_dim, output_dim, nhead=synergy_expert_nhead))
        for _ in range(num_experts - num_synergy_experts):
            self.experts.append(FeedForwardExpert(input_dim, expert_hidden_dim, output_dim))

        self.synergy_expert_indices = set(range(num_synergy_experts))

        self.router = RUSAwareGatingNetworkWithAttention(
            embedding_dim=input_dim,
            num_experts=num_experts,
            **router_config
        )

    def forward(self, x: torch.Tensor, rus_values: Dict[str, torch.Tensor], M: int, T: int) -> Tuple[torch.Tensor, Dict]:
        B, S, E_in = x.shape
        assert S == M * T and E_in == self.input_dim
        device = x.device
        num_tokens = B * S

        x_unflattened = x.view(B, M, T, E_in)
        router_logits = self.router(x_unflattened, rus_values, M, T) # (B, M, T, N_exp)
        router_logits_flat = router_logits.view(num_tokens, self.num_experts)
        gating_probs_flat = F.softmax(router_logits_flat, dim=-1) # For Aux Losses

        top_k_weights, top_k_indices = torch.topk(router_logits_flat, self.k, dim=-1)
        top_k_router_probs = F.softmax(top_k_weights, dim=-1) # Actual routing weights

        tokens_flat = x.view(num_tokens, E_in)
        final_output_flat = torch.zeros(num_tokens, self.output_dim, device=device)
        flat_expert_indices = top_k_indices.view(-1)
        flat_router_probs = top_k_router_probs.view(-1)
        flat_batch_indices = torch.arange(num_tokens, device=device).repeat_interleave(self.k)

        for expert_idx in range(self.num_experts):
            mask = (flat_expert_indices == expert_idx)
            if mask.any():
                original_token_indices = flat_batch_indices[mask]
                current_routing_probs = flat_router_probs[mask].unsqueeze(-1)
                expert_input = tokens_flat[original_token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weighted_expert_output = expert_output * current_routing_probs
                final_output_flat.index_add_(0, original_token_indices, weighted_expert_output)

        final_output = final_output_flat.view(B, S, self.output_dim)
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
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# --- TRUS-MoE Transformer Block ---
class TRUSMoEBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, moe_layer: TemporalRUSMoELayer, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_layer = moe_layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_moe = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, rus_values: Dict[str, torch.Tensor], M: int, T: int, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        moe_output, aux_moe_outputs = self.moe_layer(src, rus_values, M, T)
        src = src + self.dropout_moe(moe_output)
        src = self.norm2(src)
        return src, aux_moe_outputs


class TRUSMoEModel_LargeScale(nn.Module):
    def __init__(self,
                 input_dim: int, d_model: int, nhead: int, d_ff: int,
                 num_encoder_layers: int, num_moe_layers: int,
                 moe_config: Dict, num_modalities: int, num_classes: int,
                 dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        assert d_model == moe_config['input_dim'] and d_model == moe_config['output_dim']
        if 'synergy_expert_nhead' not in moe_config:
             moe_config['synergy_expert_nhead'] = nhead

        self.num_modalities = num_modalities
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList()

        moe_indices = set()
        if num_moe_layers > 0 and num_encoder_layers > 0:
            step = max(1, num_encoder_layers // num_moe_layers) # Ensure step is at least 1
            indices_to_add = sorted([min(i * step, num_encoder_layers - 1) for i in range(num_moe_layers)])
            # Handle potential duplicates by shifting subsequent indices
            final_indices = []
            current_idx = -1
            for idx in indices_to_add:
                chosen_idx = max(idx, current_idx + 1)
                if chosen_idx < num_encoder_layers:
                    final_indices.append(chosen_idx)
                    current_idx = chosen_idx
            moe_indices = set(final_indices)
            # Add more if needed due to boundary conditions / collisions
            while len(moe_indices) < num_moe_layers:
                 available = set(range(num_encoder_layers)) - moe_indices
                 if not available: break
                 moe_indices.add(random.choice(list(available)))

        print(f"MoE layers will be at indices: {sorted(list(moe_indices))}")

        for i in range(num_encoder_layers):
            if i in moe_indices:
                moe_layer_instance = TemporalRUSMoELayer(**moe_config)
                self.layers.append(TRUSMoEBlock(d_model, nhead, moe_layer_instance, dropout))
            else:
                self.layers.append(TransformerBlock(d_model, nhead, d_ff, dropout))

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)

    def forward(self, token_embeddings: torch.Tensor, rus_values: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Dict]]:
        B, M, T, E_in = token_embeddings.shape
        S = M * T
        x = self.input_proj(token_embeddings) * math.sqrt(self.d_model)
        x = x.view(B, S, self.d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        all_aux_moe_outputs = []
        for layer in self.layers:
            if isinstance(layer, TRUSMoEBlock):
                x, aux_outputs = layer(x, rus_values, M, T)
                all_aux_moe_outputs.append(aux_outputs)
            elif isinstance(layer, TransformerBlock):
                x = layer(x)
            else:
                raise TypeError("Unsupported layer type")

        x = self.final_norm(x)
        aggregated_output = x.mean(dim=1) # Mean pooling over sequence S
        final_logits = self.output_proj(aggregated_output)
        return final_logits, all_aux_moe_outputs


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
                    if log_probs_m1.numel() > 0: # Check if any pairs were selected
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
                    if log_probs_m1.numel() > 0:
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
                    if p_syn_m1.numel() > 0:
                        avg_p_synergy = (p_syn_m1 + p_syn_m2) / 2.0
                        neg_log_p = -torch.log(avg_p_synergy.clamp(min=epsilon)) # Clamp before log
                        sum_neglogp_synergy += neg_log_p.sum()
                        n_synergy_pairs += indicator.sum()
    L_synergy = lambda_S * (sum_neglogp_synergy / (n_synergy_pairs + epsilon))

    return L_unique, L_redundancy, L_synergy


def calculate_load_balancing_loss(gating_probs: torch.Tensor, expert_indices: torch.Tensor, k: int, lambda_load: float) -> torch.Tensor:
    """Calculates the load balancing loss (Switch Transformer version)."""
    if gating_probs.numel() == 0: return torch.tensor(0.0, device=gating_probs.device)
    B, M, T, N_exp = gating_probs.shape
    num_tokens = B * M * T
    probs_flat = gating_probs.view(num_tokens, N_exp) # P(expert|token)

    # f_i = fraction of tokens dispatched to expert i
    # Calculate how many times each expert was chosen in top-k
    expert_counts = torch.zeros(N_exp, device=gating_probs.device)
    expert_indices_flat = expert_indices.view(num_tokens, k)
    for i in range(k):
        counts = torch.bincount(expert_indices_flat[:, i], minlength=N_exp)
        expert_counts += counts
    f_i = expert_counts / num_tokens # Fraction of tokens handled by expert i

    # P_i = average router probability assigned to expert i
    P_i = probs_flat.mean(dim=0)

    # Loss = N * sum(f_i * P_i)
    load_balance_loss = N_exp * torch.sum(f_i * P_i)
    return lambda_load * load_balance_loss


class DummyMIMICDataset(Dataset):
    """
    Dummy Dataset simulating MIMIC-IV time-series data.
    Replace with actual data loading and preprocessing for MIMIC-IV.
    Crucially, assumes RUS values are pre-computed and loaded alongside data.
    """
    def __init__(self, num_samples: int, num_modalities: int, seq_len: int, input_feat_dim: int, num_classes: int):
        self.num_samples = num_samples
        self.num_modalities = num_modalities
        self.seq_len = seq_len
        self.input_feat_dim = input_feat_dim
        self.num_classes = num_classes

        # Generate random data for demonstration
        self.data = torch.randn(num_samples, num_modalities, seq_len, input_feat_dim)
        # Generate dummy RUS values
        self.rus_U = torch.rand(num_samples, num_modalities, seq_len)
        self.rus_R = torch.rand(num_samples, num_modalities, num_modalities, seq_len)
        self.rus_S = torch.rand(num_samples, num_modalities, num_modalities, seq_len)
        # Ensure symmetry for R and S
        self.rus_R = 0.5 * (self.rus_R + self.rus_R.permute(0, 2, 1, 3))
        self.rus_S = 0.5 * (self.rus_S + self.rus_S.permute(0, 2, 1, 3))
        # Generate dummy labels (binary classification)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rus_values = {
            'U': self.rus_U[idx],
            'R': self.rus_R[idx],
            'S': self.rus_S[idx]
        }
        return self.data[idx], rus_values, self.labels[idx]


def train_epoch(model: TRUSMoEModel_LargeScale,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                task_criterion: nn.Module,
                device: torch.device,
                args: argparse.Namespace):
    """Runs one training epoch."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    unique_loss_accum = 0.0
    redundancy_loss_accum = 0.0
    synergy_loss_accum = 0.0
    load_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0
    num_moe_layers_in_model = sum(isinstance(layer, TRUSMoEBlock) for layer in model.layers)

    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1}/{args.epochs} [Train]", leave=False)
    for batch_idx, (data, rus_values, labels) in enumerate(progress_bar):
        data = data.to(device)
        # Move all tensors within the rus_values dict to the device
        rus_values = {k: v.to(device) for k, v in rus_values.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        final_logits, all_aux_moe_outputs = model(data, rus_values)

        # Calculate Task Loss (assuming classification)
        task_loss = task_criterion(final_logits, labels)

        # Calculate Auxiliary Losses (sum over all MoE layers)
        total_L_unique = torch.tensor(0.0, device=device)
        total_L_redundancy = torch.tensor(0.0, device=device)
        total_L_synergy = torch.tensor(0.0, device=device)
        total_L_load = torch.tensor(0.0, device=device)
        num_moe_layers_encountered = 0

        # Iterate through model layers to find MoE blocks and get aux outputs
        moe_aux_output_index = 0
        for layer in model.layers:
             if isinstance(layer, TRUSMoEBlock):
                if moe_aux_output_index < len(all_aux_moe_outputs):
                     aux_outputs = all_aux_moe_outputs[moe_aux_output_index]
                     moe_aux_output_index += 1
                else:
                    print(f"Warning: Mismatch in aux outputs at batch {batch_idx}")
                    continue # Skip aux loss calculation for this layer if mismatch

                num_moe_layers_encountered += 1
                gating_probs = aux_outputs['gating_probs']
                expert_indices = aux_outputs['expert_indices']
                synergy_expert_indices = layer.moe_layer.synergy_expert_indices
                k = layer.moe_layer.k

                L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                    gating_probs, rus_values, synergy_expert_indices,
                    args.threshold_u, args.threshold_r, args.threshold_s,
                    args.lambda_u, args.lambda_r, args.lambda_s,
                    epsilon=args.epsilon_loss
                )
                L_load = calculate_load_balancing_loss(gating_probs, expert_indices, k, args.lambda_load)

                total_L_unique += L_unique
                total_L_redundancy += L_redundancy
                total_L_synergy += L_synergy
                total_L_load += L_load

        # Average aux losses over the number of MoE layers in this batch's forward pass
        if num_moe_layers_encountered > 0:
            total_L_unique /= num_moe_layers_encountered
            total_L_redundancy /= num_moe_layers_encountered
            total_L_synergy /= num_moe_layers_encountered
            total_L_load /= num_moe_layers_encountered

        # Combine Losses
        total_loss = task_loss + total_L_unique + total_L_redundancy + total_L_synergy + total_L_load

        # Backward pass and optimize
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN or Inf detected in total loss at batch {batch_idx}. Skipping backward pass.")
            # Optionally zero grad here anyway if desired
            # optimizer.zero_grad()
        else:
            total_loss.backward()
            # Optional: Gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            # Accumulate losses for logging
            total_loss_accum += total_loss.item()
            task_loss_accum += task_loss.item()
            unique_loss_accum += total_L_unique.item()
            redundancy_loss_accum += total_L_redundancy.item()
            synergy_loss_accum += total_L_synergy.item()
            load_loss_accum += total_L_load.item()

            # Calculate accuracy
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'TaskL': f"{task_loss.item():.4f}",
                'Acc': f"{100. * correct_predictions / total_samples:.2f}%"
            })

    # Calculate average losses and accuracy for the epoch
    avg_total_loss = total_loss_accum / len(dataloader)
    avg_task_loss = task_loss_accum / len(dataloader)
    avg_unique_loss = unique_loss_accum / len(dataloader)
    avg_redundancy_loss = redundancy_loss_accum / len(dataloader)
    avg_synergy_loss = synergy_loss_accum / len(dataloader)
    avg_load_loss = load_loss_accum / len(dataloader)
    accuracy = 100. * correct_predictions / total_samples

    print(f"Epoch {args.current_epoch+1} [Train] Avg Loss: {avg_total_loss:.4f}, Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"  Aux Losses -> Unique: {avg_unique_loss:.4f}, Redundancy: {avg_redundancy_loss:.4f}, Synergy: {avg_synergy_loss:.4f}, Load: {avg_load_loss:.4f}")

    return avg_total_loss, accuracy


def validate_epoch(model: TRUSMoEModel_LargeScale,
                   dataloader: DataLoader,
                   task_criterion: nn.Module,
                   device: torch.device,
                   args: argparse.Namespace):
    """Runs one validation epoch."""
    model.eval()
    total_loss_accum = 0.0
    task_loss_accum = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {args.current_epoch+1}/{args.epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch_idx, (data, rus_values, labels) in enumerate(progress_bar):
            data = data.to(device)
            rus_values = {k: v.to(device) for k, v in rus_values.items()}
            labels = labels.to(device)

            # Forward pass
            final_logits, all_aux_moe_outputs = model(data, rus_values)

            # Calculate Task Loss
            task_loss = task_criterion(final_logits, labels)

            # --- NOTE: Typically, auxiliary losses are not included in validation loss ---
            # ---       but could calculate them for monitoring if desired.       ---
            # total_loss = task_loss # Just use task loss for validation metric

            # Accumulate losses for logging
            # total_loss_accum += total_loss.item() # Using only task loss here
            task_loss_accum += task_loss.item()

            # Calculate accuracy
            predictions = torch.argmax(final_logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix({
                'Val TaskL': f"{task_loss.item():.4f}",
                'Val Acc': f"{100. * correct_predictions / total_samples:.2f}%"
            })

    # Calculate average losses and accuracy for the epoch
    # avg_total_loss = total_loss_accum / len(dataloader)
    avg_task_loss = task_loss_accum / len(dataloader)
    accuracy = 100. * correct_predictions / total_samples

    print(f"Epoch {args.current_epoch+1} [Val] Avg Task Loss: {avg_task_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_task_loss, accuracy


def main(args):
    """Main function to set up and run training."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Seed for reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Dummy Data Loaders ---
    # Replace with your actual MIMIC-IV data loading
    print("Setting up dummy datasets and dataloaders...")
    # Assume input_dim needs to match embedding dim E_in from data
    # For dummy data, E_in is args.input_dim
    train_dataset = DummyMIMICDataset(
        num_samples=100, # Small number for demo
        num_modalities=args.num_modalities,
        seq_len=args.seq_len,
        input_feat_dim=args.input_dim,
        num_classes=args.num_classes
    )
    val_dataset = DummyMIMICDataset(
        num_samples=20, # Small number for demo
        num_modalities=args.num_modalities,
        seq_len=args.seq_len,
        input_feat_dim=args.input_dim,
        num_classes=args.num_classes
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Dataloaders created.")

    # --- Model Configuration ---
    # MoE specific config derived from args
    moe_router_config = {
        "gru_hidden_dim": args.moe_router_gru_hidden_dim,
        "token_processed_dim": args.moe_router_token_processed_dim,
        "attn_key_dim": args.moe_router_attn_key_dim,
        "attn_value_dim": args.moe_router_attn_value_dim,
    }
    moe_layer_config = {
        "input_dim": args.d_model, # MoE layers operate at d_model dimension
        "output_dim": args.d_model,
        "num_experts": args.moe_num_experts,
        "num_synergy_experts": args.moe_num_synergy_experts,
        "k": args.moe_k,
        "expert_hidden_dim": args.moe_expert_hidden_dim,
        "synergy_expert_nhead": args.nhead, # Use same nhead for synergy expert
        "router_config": moe_router_config,
    }

    print("Initializing model...")
    model = TRUSMoEModel_LargeScale(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_encoder_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_layer_config,
        num_modalities=args.num_modalities,
        num_classes=args.num_classes,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Assuming binary classification for MIMIC-IV task (e.g., mortality)
    # If multi-class, change num_classes and potentially the loss
    task_criterion = nn.CrossEntropyLoss() # Expects logits (B, C) and labels (B,)

    print("Starting training...")
    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        args.current_epoch = epoch # Pass current epoch for logging
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, task_criterion, device, args)
        val_loss, val_acc = validate_epoch(model, val_loader, task_criterion, device, args)

        # Simple best model saving based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            print(f"Epoch {epoch+1}: New best validation accuracy: {val_acc:.2f}%.") # Placeholder saving msg


    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRUS-MoE Model on MIMIC-IV (Conceptual)')

    # Data args
    parser.add_argument('--num_modalities', type=int, default=10, help='Number of variables treated as modalities')
    parser.add_argument('--seq_len', type=int, default=48, help='Sequence length of time-series data (e.g., hours)')
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension per modality/variable (e.g., 1 if using raw values)')
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Maximum sequence length for positional encoding (M*T)')

    # Model args
    parser.add_argument('--d_model', type=int, default=128, help='Internal dimension of the model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads in MHSA and SynergyExpert')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feed-forward layer in Transformer blocks')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Total number of encoder layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of layers to replace with MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes for the task (e.g., 2 for mortality)')

    # MoE specific args
    parser.add_argument('--moe_num_experts', type=int, default=8, help='Number of experts per MoE layer')
    parser.add_argument('--moe_num_synergy_experts', type=int, default=2, help='Number of experts designated as synergy experts')
    parser.add_argument('--moe_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=128, help='Hidden dimension within experts')
    # MoE Router specific args
    parser.add_argument('--moe_router_gru_hidden_dim', type=int, default=64, help='GRU hidden dim in MoE router')
    parser.add_argument('--moe_router_token_processed_dim', type=int, default=64, help='Token processing dim in MoE router')
    parser.add_argument('--moe_router_attn_key_dim', type=int, default=32, help='Attention key dim in MoE router')
    parser.add_argument('--moe_router_attn_value_dim', type=int, default=32, help='Attention value dim in MoE router')

    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW optimizer')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Max norm for gradient clipping (0 to disable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')

    # Loss args
    parser.add_argument('--threshold_u', type=float, default=0.7, help='Threshold for uniqueness loss')
    parser.add_argument('--threshold_r', type=float, default=0.7, help='Threshold for redundancy loss')
    parser.add_argument('--threshold_s', type=float, default=0.7, help='Threshold for synergy loss')
    parser.add_argument('--lambda_u', type=float, default=0.01, help='Weight for uniqueness loss')
    parser.add_argument('--lambda_r', type=float, default=0.01, help='Weight for redundancy loss')
    parser.add_argument('--lambda_s', type=float, default=0.01, help='Weight for synergy loss')
    parser.add_argument('--lambda_load', type=float, default=0.01, help='Weight for load balancing loss')
    parser.add_argument('--epsilon_loss', type=float, default=1e-8, help='Epsilon for stability in loss calculations')

    args = parser.parse_args()
    args.current_epoch = 0
    main(args)
