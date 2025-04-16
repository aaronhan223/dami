import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RUSAwareGatingNetworkWithAttention(nn.Module):
    """
    A gating network for MoE that incorporates temporal RUS values,
    using attention to aggregate pairwise R and S information.
    """
    def __init__(self, embedding_dim, gru_hidden_dim, token_processed_dim,
                 attn_key_dim, attn_value_dim, num_experts, num_modalities):
        """
        Initializes the layers of the gating network.

        Args:
            embedding_dim (int): Dimension of the input token embeddings.
            gru_hidden_dim (int): Hidden dimension for the GRU processing RUS features.
            token_processed_dim (int): Dimension after processing token embeddings.
            attn_key_dim (int): Dimension for attention keys and queries.
            attn_value_dim (int): Dimension for attention values and output context.
            num_experts (int): The number of experts to route to.
            num_modalities (int): The number of input modalities.
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.num_experts = num_experts
        self.attn_key_dim = attn_key_dim
        self.attn_value_dim = attn_value_dim

        # Layer to process token embeddings
        self.token_processor = nn.Sequential(
            nn.Linear(embedding_dim, token_processed_dim),
            nn.ReLU()
        )

        # Linear layers for Attention mechanism
        self.query_proj = nn.Linear(token_processed_dim, attn_key_dim)
        # Pairwise feature dimension is 2 (R, S)
        self.key_proj = nn.Linear(2, attn_key_dim)
        self.value_proj = nn.Linear(2, attn_value_dim)

        # GRU to process the aggregated temporal RUS features (U, AttnContext(R,S))
        # Input dimension is 1 (U) + attn_value_dim
        self.rus_gru = nn.GRU(
            input_size=1 + attn_value_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Final MLP
        combined_dim = token_processed_dim + gru_hidden_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, num_experts)
        )

    def forward(self, token_embeddings, rus_values):
        """
        Forward pass of the gating network.

        Args:
            token_embeddings (torch.Tensor): Shape (B, M, T, E)
            rus_values (dict): Contains tensors:
                                'U': Shape (B, M, T)
                                'R': Shape (B, M, M, T)
                                'S': Shape (B, M, M, T)

        Returns:
            torch.Tensor: Routing logits. Shape (B, M, T, N_experts)
        """
        B, M, T, E = token_embeddings.shape
        if M <= 1:
             # Cannot compute pairwise features or attention if only one modality
             # Handle this edge case: maybe return simple gating based only on token?
             # Or raise error / use zero context. For simplicity, use zero context.
            processed_tokens = self.token_processor(token_embeddings) # (B, 1, T, token_proc_dim)
            # Create zero context matching GRU output dim
            rus_temporal_context = torch.zeros(B, M, T, self.rus_gru.hidden_size,
                                               device=token_embeddings.device)
            combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1)
            combined_features_flat = combined_features.view(B * M * T, -1)
            logits_flat = self.final_mlp(combined_features_flat)
            logits = logits_flat.view(B, M, T, self.num_experts)
            return logits

        U = rus_values['U'] # (B, M, T)
        R = rus_values['R'] # (B, M, M, T)
        S = rus_values['S'] # (B, M, M, T)

        # --- 1. Prepare Pairwise RUS Features ---
        # Permute R and S for easier gathering: (B, T, M, M)
        R_perm = R.permute(0, 3, 1, 2) # (B, T, M, M)
        S_perm = S.permute(0, 3, 1, 2) # (B, T, M, M)

        # --- 2. Process Token Embeddings ---
        processed_tokens = self.token_processor(token_embeddings) # (B, M, T, token_proc_dim)

        # --- 3. Compute Attention Context for R/S ---
        all_attn_contexts = []
        # Iterate over each modality 'm' to compute its specific context
        for m_idx in range(M):
            # Get token processing for this modality m: (B, T, token_proc_dim)
            query_token_repr = processed_tokens[:, m_idx, :, :]

            # Project to Query: (B, T, attn_key_dim)
            query = self.query_proj(query_token_repr)

            # Gather pairwise features F_j = (R_mj, S_mj) for j != m
            pairwise_features_list = []
            other_indices = [j for j in range(M) if j != m_idx]

            # Gather R and S values for pairs (m, j) where j != m
            # R_m_others: (B, T, M-1)
            R_m_others = R_perm[:, :, m_idx, other_indices]
            # S_m_others: (B, T, M-1)
            S_m_others = S_perm[:, :, m_idx, other_indices]

            # Stack to form pairwise features: (B, T, M-1, 2)
            pairwise_features = torch.stack([R_m_others, S_m_others], dim=-1)

            # Reshape for linear layers: (B*T*(M-1), 2)
            pairwise_features_flat = pairwise_features.reshape(-1, 2)

            # Project to Keys and Values
            keys_flat = self.key_proj(pairwise_features_flat) # (B*T*(M-1), attn_key_dim)
            values_flat = self.value_proj(pairwise_features_flat) # (B*T*(M-1), attn_value_dim)

            # Reshape Keys and Values: (B, T, M-1, Dim)
            keys = keys_flat.view(B, T, M - 1, self.attn_key_dim)
            values = values_flat.view(B, T, M - 1, self.attn_value_dim)

            # Compute Attention Scores (Scaled Dot-Product)
            # Query: (B, T, 1, Q_dim) , Keys: (B, T, K_dim, M-1) -> Scores: (B, T, 1, M-1)
            # Need shapes: Q(B, T, qd), K(B, T, M-1, kd), V(B, T, M-1, vd)
            # Attention: softmax( (Q @ K.transpose(-2,-1)) / sqrt(d_k) ) @ V
            # Query needs unsqueezing: (B, T, 1, attn_key_dim)
            query_unsqueezed = query.unsqueeze(2)

            # (B, T, 1, Q_dim) @ (B, T, K_dim, M-1) -> (B, T, 1, M-1)
            attn_scores = torch.matmul(query_unsqueezed, keys.transpose(-2, -1)) / math.sqrt(self.attn_key_dim)

            # Softmax over the 'j' dimension (M-1)
            attn_weights = F.softmax(attn_scores, dim=-1) # Shape (B, T, 1, M-1)

            # Compute weighted sum of Values
            # (B, T, 1, M-1) @ (B, T, M-1, V_dim) -> (B, T, 1, V_dim)
            attn_context = torch.matmul(attn_weights, values)

            # Squeeze the dimension corresponding to query: (B, T, attn_value_dim)
            attn_context = attn_context.squeeze(2)
            all_attn_contexts.append(attn_context)

        # Stack contexts for all modalities: (M, B, T, attn_value_dim) -> (B, M, T, attn_value_dim)
        stacked_attn_contexts = torch.stack(all_attn_contexts, dim=1)

        # --- 4. Create Attn-RUS Feature Sequence ---
        # U shape: (B, M, T) -> (B, M, T, 1)
        U_unsqueezed = U.unsqueeze(-1)
        attn_rus_features = torch.cat([U_unsqueezed, stacked_attn_contexts], dim=-1)
        # Shape: (B, M, T, 1 + attn_value_dim)

        # --- 5. Process Temporal Dynamics ---
        gru_input = attn_rus_features.view(B * M, T, -1)
        rus_temporal_context_flat, _ = self.rus_gru(gru_input)
        rus_temporal_context = rus_temporal_context_flat.view(B, M, T, -1) # (B, M, T, gru_hidden_dim)

        # --- 6. Combine Token and Temporal RUS Context ---
        combined_features = torch.cat([processed_tokens, rus_temporal_context], dim=-1)
        # Shape (B, M, T, token_processed_dim + gru_hidden_dim)

        # --- 7. Generate Routing Logits ---
        combined_features_flat = combined_features.view(B * M * T, -1)
        logits_flat = self.final_mlp(combined_features_flat)
        logits = logits_flat.view(B, M, T, self.num_experts) # Shape (B, M, T, num_experts)

        return logits