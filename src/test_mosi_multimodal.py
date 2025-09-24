import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path
import json

# Import required modules
from trus_moe_multimodal import MultimodalTRUSMoEModel
from train_mosi_multimodal import load_mosimosei_rus_data
from multibench_affect_get_data import get_dataloader
from utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
from utils.evaluation_utils import print_evaluation_results, save_evaluation_plots, save_evaluation_metrics
from plots.plot_expert_activation import analyze_expert_activations




def evaluate_model(model: MultimodalTRUSMoEModel, 
                  dataloader: DataLoader, 
                  rus_data: Dict,
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
        for batch_idx, (vision, audio, text, labels) in enumerate(progress_bar):
            batch_size = vision.shape[0]
            
            # Prepare modality data
            modality_data = [vision.to(device), audio.to(device), text.to(device)]
            
            # Prepare RUS values for this batch
            rus_values = {
                'U': torch.stack([rus_data['U'] for _ in range(batch_size)]).to(device),
                'R': torch.stack([rus_data['R'] for _ in range(batch_size)]).to(device),
                'S': torch.stack([rus_data['S'] for _ in range(batch_size)]).to(device)
            }
            labels = labels.squeeze(-1).to(device)

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
    
    # AU-ROC (for binary classification, use positive class probability)
    if all_probabilities.shape[1] == 2:
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
    else:
        # For multiclass, use one-vs-rest
        try:
            auc_score = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
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
    parser = argparse.ArgumentParser(description='Test Multimodal TRUS-MoE Model on MOSI/MOSEI Data')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset')
    parser.add_argument('--rus_data_path', type=str, required=True,
                       help='Path to the RUS data')
    parser.add_argument('--dataset', type=str, required=True, choices=['mosi', 'mosei'],
                       help='Dataset to test on (mosi or mosei)')
    
    # Optional arguments
    # parser.add_argument('--output_dir', type=str, default='../results/affect/',
    #                    help='Directory to save test results')
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--plot_expert_activations', action='store_true',
                       help='Generate expert activation plots using the test set')
    parser.add_argument('--plot_num_samples', type=int, default=32,
                       help='Number of samples from the test set to plot')
    
    # Additional evaluation options
    parser.add_argument('--eval_train', action='store_true',
                       help='Also evaluate on training set')
    parser.add_argument('--eval_val', action='store_true',
                       help='Also evaluate on validation set')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create output directory
    # args.output_dir = os.path.join(args.output_dir, args.dataset)
    args.output_dir = Path(args.checkpoint_path).parent
    # os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    model_state_dict, train_args, modality_configs, modality_names, best_val_acc = load_checkpoint(
        args.checkpoint_path, device)
    
    # Load RUS data
    print(f"Loading RUS data from {args.rus_data_path}...")
    rus_data = load_mosimosei_rus_data(args.rus_data_path, modality_names, train_args.seq_len)
    
    # Load test data
    print(f"Loading test data from {args.dataset_path}...")
    train_loader, val_loader, test_loader = get_dataloader(
        args.dataset_path, 
        data_type=args.dataset, 
        max_pad=True, 
        task='classification', 
        max_seq_len=train_args.seq_len, 
        batch_size=args.batch_size
    )
    
    # Get data info from a sample batch
    sample_batch = next(iter(test_loader))
    vision_sample, audio_sample, text_sample, labels_sample = sample_batch
    num_classes = 2
    
    print(f"Test data info:")
    print(f"  Vision shape: {vision_sample.shape}")
    print(f"  Audio shape: {audio_sample.shape}")
    print(f"  Text shape: {text_sample.shape}")
    print(f"  Labels shape: {labels_sample.shape}")
    print(f"  Number of classes: {num_classes}")
    

    # Create model
    model = create_model_from_checkpoint(model_state_dict, train_args, modality_configs, num_classes, device)
    
    # Define class names for MOSI/MOSEI (binary sentiment classification)
    class_names = ['Negative', 'Positive']
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, rus_data, device, "Test")
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
        train_results = evaluate_model(model, train_loader, rus_data, device, "Train")
        print_evaluation_results(train_results, "Train", class_names)
        
        if args.save_plots:
            save_evaluation_plots(train_results, args.output_dir, "Train", class_names)
        if args.save_metrics:
            save_evaluation_metrics(train_results, str(args.output_dir), "Train")
    
    # Optional: Evaluate on validation set
    if args.eval_val:
        print(f"\nEvaluating on validation set...")
        val_results = evaluate_model(model, val_loader, rus_data, device, "Validation")
        print_evaluation_results(val_results, "Validation", class_names)
        
        if args.save_plots:
            save_evaluation_plots(val_results, args.output_dir, "Validation", class_names)
        if args.save_metrics:
            save_evaluation_metrics(val_results, str(args.output_dir), "Validation")

    # Optional: Expert activation plots on test data
    if args.plot_expert_activations:
        print("\nGenerating expert activation plots on test data...")
        try:
            # Reinitialize a shuffled test loader with the desired batch size
            # plot_test_loader = DataLoader(
            #     dataset=test_loader.dataset,
            #     batch_size=args.plot_num_samples,
            #     shuffle=True,
            #     num_workers=args.num_workers,
            #     collate_fn=test_loader.collate_fn
            # )
            _, _, plot_test_loader = get_dataloader(
                args.dataset_path,
                data_type=args.dataset,
                max_pad=True,
                task='classification',
                max_seq_len=train_args.seq_len,
                batch_size=args.plot_num_samples
            )
            vision_b, audio_b, text_b, labels_b = next(iter(plot_test_loader))

            batch_size = vision_b.size(0)
            batch_modalities = [vision_b.to(device), audio_b.to(device), text_b.to(device)]
            batch_rus = {
                'U': torch.stack([rus_data['U'] for _ in range(batch_size)]).to(device),
                'R': torch.stack([rus_data['R'] for _ in range(batch_size)]).to(device),
                'S': torch.stack([rus_data['S'] for _ in range(batch_size)]).to(device)
            }

            plot_save_dir = os.path.join(args.output_dir, 'expert_activation_plots')
            os.makedirs(plot_save_dir, exist_ok=True)

            # Define modality names for the analyzer
            modality_names = ['vision', 'audio', 'text']

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
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
