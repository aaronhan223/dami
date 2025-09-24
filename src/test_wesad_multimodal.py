import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
from typing import Dict
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path
# Import required modules
from trus_moe_multimodal import MultimodalTRUSMoEModel
from train_wesad_multimodal import (
    MultimodalWESADDataset, 
    collate_multimodal, 
    load_wesad_rus_data
)
from wesad_rus_multimodal import load_wesad_data
from utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
from utils.evaluation_utils import print_evaluation_results, save_evaluation_plots, save_evaluation_metrics
from plots.plot_expert_activation import analyze_expert_activations


def evaluate_model(model: MultimodalTRUSMoEModel, 
                  dataloader: DataLoader, 
                  device: torch.device,
                  dataset_name: str = "Test") -> Dict:
    """
    Evaluate model on given dataset and return comprehensive metrics.
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nEvaluating on {dataset_name} set...")
    progress_bar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
    
    with torch.no_grad():
        for batch_idx, (modality_data, rus_values_batch, labels) in enumerate(progress_bar):
            # Move data to device
            modality_data = [mod.to(device) for mod in modality_data]
            rus_values = {k: v.to(device) for k, v in rus_values_batch.items()}
            labels = labels.to(device)

            # Forward pass
            final_logits, _ = model(modality_data, rus_values)

            # Calculate loss
            loss = criterion(final_logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions and probabilities
            predictions = torch.argmax(final_logits, dim=1)
            probabilities = torch.softmax(final_logits, dim=1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = accuracy_score(all_labels, all_predictions)
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_acc:.4f}"
            })
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # AU-ROC (for multiclass, use one-vs-rest)
    try:
        if all_probabilities.shape[1] > 2:
            auc_score = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
        else:
            auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
    except ValueError:
        auc_score = 0.0  # In case of issues with multiclass AUC
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Per-class metrics
    per_class_precision = precision_score(all_labels, all_predictions, average=None)
    per_class_recall = recall_score(all_labels, all_predictions, average=None)
    per_class_f1 = f1_score(all_labels, all_predictions, average=None)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'avg_loss': avg_loss,
        'confusion_matrix': cm,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test Multimodal TRUS-MoE Model on WESAD Data')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--processed_dataset_dir', type=str, required=True,
                       help='Directory containing processed WESAD dataset')
    parser.add_argument('--rus_data_path', type=str, required=True,
                       help='Path to the RUS data')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='../results/wesad/',
                       help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for DataLoader')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use. If None, will use CPU.')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    parser.add_argument('--save_metrics', action='store_true',
                       help='Save evaluation metrics to JSON files in the checkpoint directory')
    
    # Additional evaluation options
    parser.add_argument('--eval_train', action='store_true',
                       help='Also evaluate on training set')
    parser.add_argument('--eval_val', action='store_true',
                       help='Also evaluate on validation set')
    
    # Expert activation plotting args
    parser.add_argument('--plot_expert_activations', action='store_true',
                       help='Generate expert activation plots after testing')
    parser.add_argument('--plot_num_samples', type=int, default=32,
                       help='Number of samples to use for expert activation plotting')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = Path(args.checkpoint_path).parent
    
    # Load checkpoint
    model_state_dict, train_args, modality_configs, modality_names, best_val_acc = load_checkpoint(
        args.checkpoint_path, device)
    
    # Load data split information
    data_split = pickle.load(open(os.path.join(args.processed_dataset_dir, 'wesad_split.pkl'), 'rb'))
    train_subjects = data_split['train']
    val_subjects = data_split['val']
    test_subjects = data_split['test']
    
    # Load RUS data
    print(f"Loading RUS data from {args.rus_data_path}...")
    rus_data = load_wesad_rus_data(args.rus_data_path, modality_names, train_args.seq_len)
    
    # WESAD class information
    class_idx_to_name = {
        0: 'Transient',
        1: 'Baseline', 
        2: 'Stress',
        3: 'Amusement',
        4: 'Meditation',
        5: 'Unknown'
    }
    num_classes = len(class_idx_to_name)
    class_names = list(class_idx_to_name.values())
    
    print(f"WESAD dataset info:")
    print(f"  Train subjects: {train_subjects}")
    print(f"  Val subjects: {val_subjects}")
    print(f"  Test subjects: {test_subjects}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")
    print(f"  Modalities: {modality_names}")
    
    # Create model
    model = create_model_from_checkpoint(model_state_dict, train_args, modality_configs, num_classes, device)
    
    # Create test dataset and dataloader
    test_dataset = MultimodalWESADDataset(
        test_subjects, args.processed_dataset_dir, rus_data, 
        modality_names, train_args.seq_len, train_args.window_step
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_multimodal
    )
    
    print(f"Test dataset size: {len(test_dataset)}")

    # Calculate no-skill baseline accuracy on the test set
    print("Calculating no-skill baseline accuracy on the test set...")
    test_labels = []
    for _, _, labels in test_loader:
        test_labels.extend(labels.numpy())
    test_labels = np.array(test_labels)
    if len(test_labels) > 0:
        unique_lbls, counts = np.unique(test_labels, return_counts=True)
        most_frequent_class = unique_lbls[np.argmax(counts)]
        most_frequent_count = np.max(counts)
        test_baseline_acc = (most_frequent_count / len(test_labels)) * 100
        print(f"  Most frequent class (test): {class_idx_to_name.get(int(most_frequent_class), str(int(most_frequent_class)))} (class {int(most_frequent_class)})")
        print(f"  Most frequent class count (test): {int(most_frequent_count)}")
        print(f"  No-skill baseline accuracy (test): {test_baseline_acc:.2f}%")
    else:
        print("  Warning: No labels found in test loader to compute baseline accuracy.")
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device, "Test")
    print_evaluation_results(test_results, "Test", class_names)
    if args.save_metrics:
        save_evaluation_metrics(test_results, str(args.output_dir), "Test")
    
    # Save test plots
    if args.save_plots:
        save_evaluation_plots(test_results, args.output_dir, "Test", class_names)
    
    # Save predictions
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'test_predictions.npz')
        np.savez(predictions_path,
                predictions=test_results['predictions'],
                probabilities=test_results['probabilities'],
                labels=test_results['labels'])
        print(f"Predictions saved to {predictions_path}")
    
    # Optional: Evaluate on training set
    if args.eval_train:
        print(f"\nEvaluating on training set...")
        train_dataset = MultimodalWESADDataset(
            train_subjects, args.processed_dataset_dir, rus_data,
            modality_names, train_args.seq_len, train_args.window_step
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=collate_multimodal
        )
        
        train_results = evaluate_model(model, train_loader, device, "Train")
        print_evaluation_results(train_results, "Train", class_names)
        
        if args.save_plots:
            save_evaluation_plots(train_results, args.output_dir, "Train", class_names)
        if args.save_metrics:
            save_evaluation_metrics(train_results, str(args.output_dir), "Train")
    
    # Optional: Evaluate on validation set
    if args.eval_val:
        print(f"\nEvaluating on validation set...")
        val_dataset = MultimodalWESADDataset(
            val_subjects, args.processed_dataset_dir, rus_data,
            modality_names, train_args.seq_len, train_args.window_step
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=collate_multimodal
        )
        
        val_results = evaluate_model(model, val_loader, device, "Validation")
        print_evaluation_results(val_results, "Validation", class_names)
        
        if args.save_plots:
            save_evaluation_plots(val_results, args.output_dir, "Validation", class_names)
        if args.save_metrics:
            save_evaluation_metrics(val_results, str(args.output_dir), "Validation")
    
    # Generate expert activation plots if requested
    if args.plot_expert_activations:
        print("\nGenerating expert activation plots for the test model...")
        
        # Create a temporary dataloader with the desired batch size for plotting
        plot_loader = DataLoader(
            test_dataset,
            batch_size=args.plot_num_samples,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=collate_multimodal
        )
        
        plot_iter = iter(plot_loader)
        batch_modalities, batch_rus, batch_labels = next(plot_iter)
        # Move to device (same as evaluation loops)
        batch_modalities = [mod.to(device) for mod in batch_modalities]
        batch_rus = {k: v.to(device) for k, v in batch_rus.items()}
        
        # Generate plots
        plot_save_dir = os.path.join(args.output_dir, 'expert_activation_plots_test')
        
        try:
            analyze_expert_activations(
                time_moe_model=model,
                baseline_model=None,
                data_batch=batch_modalities,
                rus_values=batch_rus,
                modality_names=modality_names,
                save_dir=plot_save_dir,
                moe_num_synergy_experts=train_args.moe_num_synergy_experts
            )
            print(f"Expert activation plots saved to {plot_save_dir}")
        except Exception as e:
            print(f"Error generating expert activation plots: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Best training validation accuracy: {best_val_acc:.4f}")
    print(f"Test AU-ROC: {test_results['auc_score']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
