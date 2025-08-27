#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple
import math
import random
from tqdm import tqdm
import wandb
import pdb

# Import TRUS MoE components
from trus_moe_multimodal import (
    MultimodalTRUSMoEModel, 
    ModalitySpecificEncoder,
    MultimodalTemporalRUSMoELayer
)
from trus_moe_model import calculate_rus_losses, calculate_load_balancing_loss
from temporal_rus_label import temporal_pid_label_multi_sequence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def drop_empty_entries(dataset: Dict) -> Dict:
    """Drop entries where there's no text data (following i2moe approach)"""
    drop_indices = []
    for idx, text_seq in enumerate(dataset["text"]):
        if text_seq.sum() == 0:
            drop_indices.append(idx)
    
    if drop_indices:
        logger.info(f"Dropping {len(drop_indices)} entries with empty text")
        for modality in dataset.keys():
            dataset[modality] = np.delete(dataset[modality], drop_indices, 0)
    
    return dataset

def load_mosi_data(data_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load MOSI dataset from pickle file (following i2moe data format)"""
    logger.info(f"Loading MOSI data from {data_path}")
    
    with open(data_path, "rb") as f:
        alldata = pickle.load(f)
    
    # Clean data by dropping empty entries
    alldata["train"] = drop_empty_entries(alldata["train"])
    alldata["valid"] = drop_empty_entries(alldata["valid"])  
    alldata["test"] = drop_empty_entries(alldata["test"])
    pdb.set_trace()
    logger.info(f"Data shapes - Train: {len(alldata['train']['labels'])}, "
                f"Valid: {len(alldata['valid']['labels'])}, "
                f"Test: {len(alldata['test']['labels'])}")
    
    return alldata["train"], alldata["valid"], alldata["test"]

def preprocess_mosi_data(train_data: Dict, valid_data: Dict, test_data: Dict, 
                        task_type: str = "classification") -> Tuple[List[torch.Tensor], ...]:
    """
    Preprocess MOSI data for multimodal TRUS MoE training.
    
    Args:
        train_data, valid_data, test_data: Raw data dicts with keys ['vision', 'audio', 'text', 'labels', 'id']
        task_type: "classification" or "regression"
    
    Returns:
        Processed tensors for training
    """
    logger.info("Preprocessing MOSI data...")
    
    def process_split(data_dict):
        # Get modality data and handle NaN values (following i2moe approach)
        vision_data = []
        for i in range(len(data_dict["vision"])):
            vision_seq = np.nan_to_num(data_dict["vision"][i])
            vision_data.append(vision_seq)
        vision_data = np.array(vision_data)
        
        audio_data = []
        for i in range(len(data_dict["audio"])):
            audio_seq = np.nan_to_num(data_dict["audio"][i])
            audio_data.append(audio_seq)
        audio_data = np.array(audio_data)
        
        text_data = []
        for i in range(len(data_dict["text"])):
            text_seq = np.nan_to_num(data_dict["text"][i])
            text_data.append(text_seq)
        text_data = np.array(text_data)
        
        # Process labels
        labels = data_dict["labels"].flatten()
        if task_type == "classification":
            # Convert to binary classification: negative (<=0) vs positive (>0)
            labels = np.array([0 if label <= 0 else 1 for label in labels])
        
        return vision_data, audio_data, text_data, labels
    
    # Process each split
    train_vision, train_audio, train_text, train_labels = process_split(train_data)
    valid_vision, valid_audio, valid_text, valid_labels = process_split(valid_data) 
    test_vision, test_audio, test_text, test_labels = process_split(test_data)
    
    # Convert to tensors
    train_tensors = [
        torch.FloatTensor(train_vision),
        torch.FloatTensor(train_audio), 
        torch.FloatTensor(train_text),
        torch.LongTensor(train_labels) if task_type == "classification" else torch.FloatTensor(train_labels)
    ]
    
    valid_tensors = [
        torch.FloatTensor(valid_vision),
        torch.FloatTensor(valid_audio),
        torch.FloatTensor(valid_text), 
        torch.LongTensor(valid_labels) if task_type == "classification" else torch.FloatTensor(valid_labels)
    ]
    
    test_tensors = [
        torch.FloatTensor(test_vision),
        torch.FloatTensor(test_audio),
        torch.FloatTensor(test_text),
        torch.LongTensor(test_labels) if task_type == "classification" else torch.FloatTensor(test_labels)
    ]
    
    logger.info(f"Processed data shapes:")
    logger.info(f"  Vision: {train_vision.shape}, Audio: {train_audio.shape}, Text: {train_text.shape}")
    logger.info(f"  Input dimensions - Vision: {train_vision.shape[-1]}, "
                f"Audio: {train_audio.shape[-1]}, Text: {train_text.shape[-1]}")
    
    return train_tensors, valid_tensors, test_tensors

def compute_temporal_correlations(seq1: np.ndarray, seq2: np.ndarray, max_lag: int = 5) -> float:
    """Compute temporal correlation with lags"""
    max_corr = 0.0
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(seq1, seq2)[0, 1]
        else:
            if len(seq1) > lag:
                corr = np.corrcoef(seq1[:-lag], seq2[lag:])[0, 1]
            else:
                corr = 0.0
        corr = np.nan_to_num(corr, 0.0)
        max_corr = max(max_corr, abs(corr))
    return max_corr

def compute_temporal_uniqueness(sequence: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute temporal uniqueness using sliding window variance"""
    T = len(sequence)
    uniqueness = np.zeros(T)
    
    for t in range(T):
        # Define window around current time point
        start = max(0, t - window_size // 2)
        end = min(T, t + window_size // 2 + 1)
        window = sequence[start:end]
        
        # Compute local variance as measure of uniqueness
        local_var = np.var(window)
        # Compute change from previous time step
        if t > 0:
            temporal_change = abs(sequence[t] - sequence[t-1])
        else:
            temporal_change = 0.0
            
        # Combine local variance and temporal change
        uniqueness[t] = local_var + 0.1 * temporal_change
        
    return uniqueness

def compute_rus_values(modality_tensors: List[torch.Tensor], device: torch.device, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Compute temporal RUS values that properly account for temporal dependencies.
    
    Args:
        modality_tensors: List of [vision, audio, text] tensors, each (B, T, D)
        device: torch device
        labels: Optional tensor of labels (B,) for classification context
    
    Returns:
        Dictionary with temporal RUS tensors
    """
    B, T = modality_tensors[0].shape[:2]
    M = len(modality_tensors)  # 3 modalities: vision, audio, text
    
    # Initialize RUS tensors
    U = torch.zeros(B, M, T, device=device)
    R = torch.zeros(B, M, M, T, device=device) 
    S = torch.zeros(B, M, M, T, device=device)
    
    try:
        # Compute temporal RUS for each sample in the batch
        for b in range(B):
            # Extract time series for this sample - aggregate across feature dimensions
            modality_series = []
            for m in range(M):
                # Average across feature dimension to get a single time series per modality
                series = modality_tensors[m][b, :, :].mean(dim=-1).cpu().numpy()  # (T,)
                modality_series.append(series)
            
            # Compute temporal uniqueness for each modality
            for m in range(M):
                uniqueness_values = compute_temporal_uniqueness(modality_series[m], window_size=min(5, T//2))
                U[b, m, :] = torch.tensor(uniqueness_values, device=device)
            
            # Compute temporal redundancy between modality pairs
            for m1 in range(M):
                for m2 in range(M):
                    if m1 != m2:
                        # Use sliding window to compute time-varying redundancy
                        window_size = min(10, T//2)
                        redundancy_values = np.zeros(T)
                        
                        for t in range(T):
                            # Define temporal window
                            start = max(0, t - window_size // 2)
                            end = min(T, t + window_size // 2 + 1)
                            
                            # Extract window data
                            window1 = modality_series[m1][start:end]
                            window2 = modality_series[m2][start:end]
                            
                            # Compute temporal correlation within window
                            temporal_corr = compute_temporal_correlations(window1, window2, max_lag=3)
                            redundancy_values[t] = temporal_corr
                        
                        R[b, m1, m2, :] = torch.tensor(redundancy_values, device=device)
            
            # Compute temporal synergy (interactions that require all modalities)
            if M >= 3:
                window_size = min(8, T//2)
                
                for m1 in range(M):
                    for m2 in range(m1 + 1, M):
                        synergy_values = np.zeros(T)
                        
                        for t in range(T):
                            # Define temporal window
                            start = max(0, t - window_size // 2)
                            end = min(T, t + window_size // 2 + 1)
                            
                            # Extract window data for all three modalities
                            windows = [modality_series[m][start:end] for m in range(M)]
                            
                            # Compute synergy as joint information beyond pairwise
                            pairwise_corr = 0.0
                            joint_pattern = 0.0
                            
                            # Average pairwise correlations
                            for i in range(M):
                                for j in range(i + 1, M):
                                    corr = compute_temporal_correlations(windows[i], windows[j], max_lag=2)
                                    pairwise_corr += corr
                            pairwise_corr /= (M * (M - 1) / 2)
                            
                            # Joint pattern: how much all modalities co-vary together
                            if len(windows[0]) > 1:
                                # Compute how well the sum of all modalities predicts individual modalities
                                joint_signal = np.sum([w for w in windows], axis=0) / M
                                joint_correlations = []
                                for w in windows:
                                    corr = compute_temporal_correlations(joint_signal, w, max_lag=1)
                                    joint_correlations.append(corr)
                                joint_pattern = np.mean(joint_correlations)
                            
                            # Synergy is joint pattern beyond simple pairwise redundancy
                            synergy_values[t] = max(0, joint_pattern - pairwise_corr * 0.5)
                        
                        S[b, m1, m2, :] = torch.tensor(synergy_values, device=device)
                        S[b, m2, m1, :] = S[b, m1, m2, :]  # Symmetric
    
    except Exception as e:
        logger.warning(f"Error computing temporal RUS values: {e}, using fallback values")
        # Use temporally structured fallback values
        for b in range(B):
            # Create temporal patterns with some structure
            t_indices = torch.arange(T, device=device, dtype=torch.float)
            
            # Uniqueness: varies over time with some randomness
            for m in range(M):
                base_pattern = 0.3 * torch.sin(2 * np.pi * t_indices / T) + 0.2
                noise = 0.1 * torch.randn(T, device=device)
                U[b, m, :] = torch.clamp(base_pattern + noise, 0.01, 0.99)
            
            # Redundancy: also time-varying
            for m1 in range(M):
                for m2 in range(M):
                    if m1 != m2:
                        base_pattern = 0.2 * torch.cos(2 * np.pi * t_indices / T) + 0.15
                        noise = 0.05 * torch.randn(T, device=device)
                        R[b, m1, m2, :] = torch.clamp(base_pattern + noise, 0.01, 0.5)
            
            # Synergy: lower magnitude, time-varying
            for m1 in range(M):
                for m2 in range(m1 + 1, M):
                    base_pattern = 0.1 * torch.sin(4 * np.pi * t_indices / T) + 0.05
                    noise = 0.02 * torch.randn(T, device=device)
                    S[b, m1, m2, :] = torch.clamp(base_pattern + noise, 0.01, 0.3)
                    S[b, m2, m1, :] = S[b, m1, m2, :]
    
    # Ensure all values are in reasonable ranges
    U = torch.clamp(U, 0.01, 1.0)
    R = torch.clamp(R, 0.01, 1.0)
    S = torch.clamp(S, 0.01, 1.0)
    
    rus_values = {'U': U, 'R': R, 'S': S}
    
    return rus_values

def create_data_loaders(train_tensors: List[torch.Tensor], 
                       valid_tensors: List[torch.Tensor],
                       test_tensors: List[torch.Tensor], 
                       batch_size: int, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training"""
    
    # Create datasets
    train_dataset = TensorDataset(*train_tensors)
    valid_dataset = TensorDataset(*valid_tensors)
    test_dataset = TensorDataset(*test_tensors)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, valid_loader, test_loader

def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device, task_type: str = "classification") -> Dict:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            vision, audio, text, labels = [x.to(device) for x in batch]
            
            # Prepare modality inputs
            modality_inputs = [vision, audio, text]
            
            # Compute RUS values for this batch
            rus_values = compute_rus_values(modality_inputs, device)
            
            # Forward pass
            logits, _ = model(modality_inputs, rus_values)
            
            # Compute loss
            if task_type == "classification":
                loss = criterion(logits, labels)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities
            else:
                loss = criterion(logits.squeeze(), labels)
                preds = logits.squeeze()
                all_probs.extend(preds.cpu().numpy())
                
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    if task_type == "classification":
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        auc = roc_auc_score(all_labels, all_probs)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }
    else:
        # For regression, convert predictions to binary for accuracy
        binary_preds = (np.array(all_preds) > 0).astype(int)
        binary_labels = (np.array(all_labels) > 0).astype(int)
        accuracy = accuracy_score(binary_labels, binary_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'mae': np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
        }

def train_model(args):
    """Main training function"""
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_config = {k: v for k, v in vars(args).items()}
        run_name = f"mosi_{args.task_type}_seq{args.d_model}"
        if args.wandb_run_name:
            run_name = args.wandb_run_name
            
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=wandb_config,
            name=run_name,
            mode="online" if not args.wandb_disabled else "disabled"
        )
    
    # Set device and seed
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    set_seed(args.seed)
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    train_data, valid_data, test_data = load_mosi_data(args.data_path)
    train_tensors, valid_tensors, test_tensors = preprocess_mosi_data(
        train_data, valid_data, test_data, args.task_type
    )
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_tensors, valid_tensors, test_tensors, args.batch_size, args.num_workers
    )
    
    # Model configuration following i2moe dimensions
    modality_configs = [
        {'input_dim': 20, 'num_layers': 2, 'use_cnn': False},  # Vision
        {'input_dim': 5, 'num_layers': 2, 'use_cnn': False},   # Audio  
        {'input_dim': 300, 'num_layers': 2, 'use_cnn': False}  # Text
    ]
    
    moe_config = {
        'num_experts': args.num_experts,
        'num_synergy_experts': args.num_synergy_experts, 
        'k': args.top_k,
        'expert_hidden_dim': args.expert_hidden_dim,
        'synergy_expert_nhead': args.synergy_nhead,
        'router_config': {
            'gru_hidden_dim': args.gru_hidden_dim,
            'token_processed_dim': args.token_processed_dim,
            'attn_key_dim': args.attn_key_dim,
            'attn_value_dim': args.attn_value_dim
        },
        'use_load_balancing': args.use_load_balancing,
        'capacity_factor': args.capacity_factor,
        'drop_tokens': args.drop_tokens
    }
    
    num_classes = 2 if args.task_type == "classification" else 1
    
    # Initialize model
    model = MultimodalTRUSMoEModel(
        modality_configs=modality_configs,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_layers,
        num_moe_layers=args.num_moe_layers,
        moe_config=moe_config,
        num_classes=num_classes,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    if args.task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.SmoothL1Loss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_metric = 0.0 if args.task_type == "classification" else float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_rus_loss = 0.0
        total_load_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            vision, audio, text, labels = [x.to(device) for x in batch]
            pdb.set_trace()
            # Prepare modality inputs
            modality_inputs = [vision, audio, text]
            
            # Compute RUS values for this batch
            rus_values = compute_rus_values(modality_inputs, device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, aux_outputs = model(modality_inputs, rus_values)
            
            # Task loss
            if args.task_type == "classification":
                task_loss = criterion(logits, labels)
            else:
                task_loss = criterion(logits.squeeze(), labels)
            
            # Auxiliary losses
            rus_loss = torch.tensor(0.0, device=device)
            load_loss = torch.tensor(0.0, device=device)
            
            if aux_outputs:  # Only compute if we have MoE outputs
                for aux_output in aux_outputs:
                    gating_probs = aux_output['gating_probs']
                    expert_indices = aux_output['expert_indices']
                    
                    # RUS losses
                    if args.lambda_rus > 0:
                        L_unique, L_redundancy, L_synergy = calculate_rus_losses(
                            gating_probs, rus_values, 
                            set(range(args.num_synergy_experts)),
                            threshold_U=args.threshold_U,
                            threshold_R=args.threshold_R, 
                            threshold_S=args.threshold_S,
                            lambda_U=1.0, lambda_R=1.0, lambda_S=1.0
                        )
                        rus_loss += L_unique + L_redundancy + L_synergy
                    
                    # Load balancing loss
                    if args.lambda_load > 0 and args.use_load_balancing:
                        load_loss += calculate_load_balancing_loss(
                            gating_probs, expert_indices, args.top_k, 1.0
                        )
            
            # Total loss
            total_batch_loss = (task_loss + 
                              args.lambda_rus * rus_loss + 
                              args.lambda_load * load_loss)
            
            total_batch_loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Update statistics
            total_loss += total_batch_loss.item()
            total_task_loss += task_loss.item()
            total_rus_loss += rus_loss.item()
            total_load_loss += load_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total': f'{total_batch_loss.item():.4f}',
                'Task': f'{task_loss.item():.4f}', 
                'RUS': f'{rus_loss.item():.4f}',
                'Load': f'{load_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # Log training statistics
        avg_loss = total_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)
        avg_rus_loss = total_rus_loss / len(train_loader)
        avg_load_loss = total_load_loss / len(train_loader)
        
        logger.info(f"Training - Total: {avg_loss:.4f}, Task: {avg_task_loss:.4f}, "
                   f"RUS: {avg_rus_loss:.4f}, Load: {avg_load_loss:.4f}")
        
        # Log training metrics to wandb
        if args.use_wandb:
            wandb.log({
                "train/total_loss": avg_loss,
                "train/task_loss": avg_task_loss,
                "train/rus_loss": avg_rus_loss,
                "train/load_balancing_loss": avg_load_loss,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            })
        
        # Validation
        if (epoch + 1) % args.eval_freq == 0:
            val_results = evaluate_model(model, valid_loader, criterion, device, args.task_type)
            
            if args.task_type == "classification":
                val_metric = val_results['accuracy']
                logger.info(f"Validation - Loss: {val_results['loss']:.4f}, "
                           f"Acc: {val_results['accuracy']:.4f}, "
                           f"F1: {val_results['f1']:.4f}, AUC: {val_results['auc']:.4f}")
                
                # Log validation metrics to wandb
                if args.use_wandb:
                    wandb.log({
                        "val/loss": val_results['loss'],
                        "val/accuracy": val_results['accuracy'],
                        "val/f1": val_results['f1'],
                        "val/auc": val_results['auc'],
                        "epoch": epoch + 1
                    })
            else:
                val_metric = -val_results['loss']  # Higher is better for minimizing loss
                logger.info(f"Validation - Loss: {val_results['loss']:.4f}, "
                           f"Acc: {val_results['accuracy']:.4f}, MAE: {val_results['mae']:.4f}")
                
                # Log validation metrics to wandb
                if args.use_wandb:
                    wandb.log({
                        "val/loss": val_results['loss'],
                        "val/accuracy": val_results['accuracy'],
                        "val/mae": val_results['mae'],
                        "epoch": epoch + 1
                    })
            
            # Early stopping and model saving
            if ((args.task_type == "classification" and val_metric > best_metric) or
                (args.task_type == "regression" and val_metric > best_metric)):
                best_metric = val_metric
                patience_counter = 0
                
                # Save best model
                if args.save_path:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_metric': best_metric,
                        'args': args
                    }, args.save_path)
                    logger.info(f"Saved best model with metric: {best_metric:.4f}")
                    
                    # Log best metric to wandb
                    if args.use_wandb:
                        if args.task_type == "classification":
                            wandb.log({"best_val_accuracy": best_metric, "best_epoch": epoch + 1})
                        else:
                            wandb.log({"best_val_metric": best_metric, "best_epoch": epoch + 1})
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    # Final evaluation on test set
    if args.save_path and os.path.exists(args.save_path):
        checkpoint = torch.load(args.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model for testing")
    
    test_results = evaluate_model(model, test_loader, criterion, device, args.task_type)
    
    if args.task_type == "classification":
        logger.info(f"Test Results - Loss: {test_results['loss']:.4f}, "
                   f"Acc: {test_results['accuracy']:.4f}, "
                   f"F1: {test_results['f1']:.4f}, AUC: {test_results['auc']:.4f}")
        
        # Log test results to wandb
        if args.use_wandb:
            wandb.log({
                "test/loss": test_results['loss'],
                "test/accuracy": test_results['accuracy'],
                "test/f1": test_results['f1'],
                "test/auc": test_results['auc']
            })
    else:
        logger.info(f"Test Results - Loss: {test_results['loss']:.4f}, "
                   f"Acc: {test_results['accuracy']:.4f}, MAE: {test_results['mae']:.4f}")
        
        # Log test results to wandb
        if args.use_wandb:
            wandb.log({
                "test/loss": test_results['loss'],
                "test/accuracy": test_results['accuracy'],
                "test/mae": test_results['mae']
            })
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train TRUS MoE on MOSI Dataset')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='/cis/home/xhan56/MOSI/mosi_data_0610.pkl',
                       help='Path to MOSI pickle data file')
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression'],
                       default='classification', help='Task type')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='Number of MoE layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_len', type=int, default=1000, help='Maximum sequence length')
    
    # MoE arguments
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--num_synergy_experts', type=int, default=2, help='Number of synergy experts')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k routing')
    parser.add_argument('--expert_hidden_dim', type=int, default=512, help='Expert hidden dimension')
    parser.add_argument('--synergy_nhead', type=int, default=4, help='Synergy expert attention heads')
    
    # Router arguments
    parser.add_argument('--gru_hidden_dim', type=int, default=128, help='Router GRU hidden dim')
    parser.add_argument('--token_processed_dim', type=int, default=128, help='Token processing dim')
    parser.add_argument('--attn_key_dim', type=int, default=64, help='Attention key dimension')
    parser.add_argument('--attn_value_dim', type=int, default=64, help='Attention value dimension')
    
    # RUS loss arguments
    parser.add_argument('--lambda_rus', type=float, default=0.1, help='RUS loss weight')
    parser.add_argument('--lambda_load', type=float, default=0.01, help='Load balancing loss weight')
    parser.add_argument('--threshold_U', type=float, default=0.1, help='Uniqueness threshold')
    parser.add_argument('--threshold_R', type=float, default=0.1, help='Redundancy threshold')
    parser.add_argument('--threshold_S', type=float, default=0.1, help='Synergy threshold')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency')
    
    # System arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--use_load_balancing', action='store_true', help='Use load balancing loss')
    parser.add_argument('--capacity_factor', type=float, default=1.25, help='MoE capacity factor')
    parser.add_argument('--drop_tokens', action='store_true', help='Drop overflow tokens')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, default=None, help='Model save path')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mosi-trus-moe', 
                       help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/username')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_disabled', action='store_true', help='Disable wandb')
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f'mosi_train_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set default save path if not provided
    if args.save_path is None:
        args.save_path = os.path.join(args.log_dir, f'best_model_{timestamp}.pth')
    
    logger.info("Starting MOSI training with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Start training
    train_model(args)

if __name__ == '__main__':
    main()