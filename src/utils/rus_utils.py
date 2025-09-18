"""
RUS (Redundancy, Uniqueness, Synergy) utility functions for multimodal training.
"""

import torch
from typing import Dict


def extend_rus_with_sequence_length(rus_batches: Dict[str, torch.Tensor], seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Extend RUS batches to include temporal dimension by copying values.
    
    Args:
        rus_batches: Dict with 'U', 'R', 'S' tensors
        seq_len: Sequence length (T)
    
    Returns:
        Extended rus_batches with temporal dimension
    """
    extended_rus = {}
    
    # U: (B, M) -> (B, M, T)
    extended_rus['U'] = rus_batches['U'].unsqueeze(-1).expand(-1, -1, seq_len)
    
    # R: (B, M, M) -> (B, M, M, T)  
    extended_rus['R'] = rus_batches['R'].unsqueeze(-1).expand(-1, -1, -1, seq_len)
    
    # S: (B, M, M) -> (B, M, M, T)
    extended_rus['S'] = rus_batches['S'].unsqueeze(-1).expand(-1, -1, -1, seq_len)
    
    return extended_rus
