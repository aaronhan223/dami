

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
import pickle
import numpy as np
import itertools
import torch
from temporal_pid_multivariate import multi_lag_analysis
from wesad_rus_multimodal import load_wesad_data, multi_lag_analysis_multi_subject


BODY_PART_MODALITIES = {
    'chest': {
        'sources': [
            'chest_ecg',
            'chest_eda',
            'chest_emg',
            'chest_resp',
            'chest_acc'
        ],
        'target': 'chest_temp'
    },
    'wrist': {
        'sources': [
            'wrist_bvp',
            'wrist_eda',
            'wrist_acc',
        ],
        'target': 'wrist_temp'
    }
}

MODALITY_COLUMNS = {
    'chest_ecg': ['chest_ECG_0'],
    'chest_eda': ['chest_EDA_0'],
    'chest_emg': ['chest_EMG_0'],
    'chest_resp': ['chest_Resp_0'],
    'chest_acc': ['chest_ACC_0', 'chest_ACC_1', 'chest_ACC_2'],
    'chest_temp': ['chest_Temp_0'],
    'wrist_bvp': ['wrist_BVP_0'],
    'wrist_eda': ['wrist_EDA_0'],
    'wrist_acc': ['wrist_ACC_0', 'wrist_ACC_1', 'wrist_ACC_2'],
    'wrist_temp': ['wrist_TEMP_0'],
}



def extract_modality_data(subject_data, modality_name):
    """Extract data for a specific fine-grained modality from subject data."""
    columns = MODALITY_COLUMNS[modality_name]
    
    # Get all available columns from both chest and wrist
    all_chest_columns = subject_data['chest_columns']
    all_wrist_columns = subject_data['wrist_columns']
    
    modality_data = []
    
    for col in columns:
        if col in all_chest_columns:
            col_idx = all_chest_columns.index(col)
            modality_data.append(subject_data['chest_signals'][:, col_idx])
        elif col in all_wrist_columns:
            col_idx = all_wrist_columns.index(col)
            modality_data.append(subject_data['wrist_signals'][:, col_idx])
        else:
            raise ValueError(f"Column {col} not found in chest or wrist columns")
    
    result = np.column_stack(modality_data)
    
    # If only one column, squeeze to 1D array
    if result.shape[1] == 1:
        result = result.squeeze(axis=1)
    
    return result


def main(args):
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    # Load data split
    data_split = pickle.load(open(os.path.join(args.processed_dataset_dir, 'wesad_split.pkl'), 'rb'))
    # train_subjects = data_split['train']
    # print(f"Data split train: {data_split['train'][0]}")
    subject = data_split['train'][0]
    print(f"Subject: {subject}")
    
    # Load subject data
    # train_subjects_data = load_wesad_data(train_subjects, args.processed_dataset_dir)
    subject_data = load_wesad_data([subject], args.processed_dataset_dir)[0]
    
    # Generate all pairs of fine-grained modalities
    print(f"Analyzing body part: {args.body_part}")
    modality_pairs = list(itertools.combinations(BODY_PART_MODALITIES[args.body_part]['sources'], 2))
    target_modality = BODY_PART_MODALITIES[args.body_part]['target']

    print(f"Modality pairs: {modality_pairs}")
    print(f"Target modality: {target_modality}")

    all_pid_results = []

    for mod1, mod2 in modality_pairs:
        print(f"\n--- Analyzing Fine-Grained Modality Pair {mod1} vs {mod2} ---")
        
        # Extract data for each modality across all subjects
        # X1_list = []
        # X2_list = []
        # Y_list = []
        
        # for subject_data in subject_data:
        #     try:
        #         X1_data = extract_modality_data(subject_data, mod1)
        #         # print(f"Modality {mod1} X1_data shape: {X1_data.shape}")
        #         X2_data = extract_modality_data(subject_data, mod2)
        #         # print(f"Modality {mod2} X2_data shape: {X2_data.shape}")
        #         Y_data = extract_modality_data(subject_data, target_modality)
        #         # print(f"Target modality {target_modality} Y_data shape: {Y_data.shape}")
                
        #         X1_list.append(X1_data)
        #         X2_list.append(X2_data)
        #         Y_list.append(Y_data)
                
                
        #     except Exception as e:
        #         print(f"Warning: Error extracting data for subject: {e}")
        #         continue
        
        # if not X1_list:
        #     print(f"No valid data found for pair ({mod1}, {mod2})")
        #     continue
        
        # print(f"  {mod1} shape: {X1_list[0].shape}, {mod2} shape: {X2_list[0].shape}")
        
        # Perform PID analysis
        # pid_results = multi_lag_analysis_multi_subject(X1_list, X2_list, Y_list,
        #                                                args.max_lag, args.bins,
        #                                                method='batch',
        #                                                batch_size=args.batch_size,
        #                                                n_batches=args.n_batches,
        #                                                seed=args.seed,
        #                                                device=device,
        #                                                hidden_dim=args.hidden_dim,
        #                                                layers=args.layers,
        #                                                activation=args.activation,
        #                                                lr=args.lr,
        #                                                embed_dim=args.embed_dim,
        #                                                discrim_epochs=args.discrim_epochs,
        #                                                ce_epochs=args.ce_epochs)

        X1 = extract_modality_data(subject_data, mod1)
        X2 = extract_modality_data(subject_data, mod2)
        Y = extract_modality_data(subject_data, target_modality)
        
        if len(X1) != len(Y) or len(X2) != len(Y):
            print(f"Warning: Length mismatch for pair ({mod1}, {mod2}). Skipping.")
            print(f"Len X1: {len(X1)}, Len X2: {len(X2)}, Len Y: {len(Y)}")
            continue

        pid_results = multi_lag_analysis(X1, X2, Y, max_lag=args.max_lag, bins=args.bins, method='joint')
        
        lags = pid_results.get('lag', range(args.max_lag + 1))
        lag_results = []

        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx]
                u2 = pid_results['unique_x2'][lag_idx]
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                if mi > 1e-9:
                    lag_results.append({
                        'lag': lag,
                        'R_value': r,
                        'U1_value': u1,
                        'U2_value': u2,
                        'S_value': s,
                        'MI_value': mi,
                        'R_norm': r / mi,
                        'U1_norm': u1 / mi,
                        'U2_norm': u2 / mi,
                        'S_norm': s / mi
                    })
            except (IndexError, KeyError) as e:
                print(f"Warning: Error processing lag {lag} for pair ({mod1}, {mod2}): {e}")
                continue
        
        if lag_results:
            # Calculate average metrics across all lags
            avg_metrics = {
                'R_value': np.mean([r['R_value'] for r in lag_results]),
                'U1_value': np.mean([r['U1_value'] for r in lag_results]),
                'U2_value': np.mean([r['U2_value'] for r in lag_results]),
                'S_value': np.mean([r['S_value'] for r in lag_results]),
                'MI_value': np.mean([r['MI_value'] for r in lag_results]),
                'R_norm': np.mean([r['R_norm'] for r in lag_results]),
                'U1_norm': np.mean([r['U1_norm'] for r in lag_results]),
                'U2_norm': np.mean([r['U2_norm'] for r in lag_results]),
                'S_norm': np.mean([r['S_norm'] for r in lag_results])
            }

            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': avg_metrics,
                'lag_results': lag_results,
                'n_features_mod1': len(MODALITY_COLUMNS[mod1]),
                'n_features_mod2': len(MODALITY_COLUMNS[mod2]),
                'modality_columns': {
                    'mod1_columns': MODALITY_COLUMNS[mod1],
                    'mod2_columns': MODALITY_COLUMNS[mod2]
                }
            })
            
            print(f"  Completed analysis for {mod1} vs {mod2}")
            print(f"    Average R: {avg_metrics['R_norm']:.4f}, U1: {avg_metrics['U1_norm']:.4f}, "
                  f"U2: {avg_metrics['U2_norm']:.4f}, S: {avg_metrics['S_norm']:.4f}")
    
    # Save results
    if all_pid_results:
        all_output_filename = f'rus_fine_grained_{args.body_part}_modality_lag{args.max_lag}_bins{args.bins}.npy'
        all_output_path = os.path.join(args.output_dir, all_output_filename)
        print(f"\nSaving {len(all_pid_results)} PID results for fine-grained modality pairs to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True)
        print("Saving fine-grained modality analysis complete.")
    
    print(f"\nAnalysis complete for all {len(modality_pairs)} fine-grained modality pairs.")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for result in all_pid_results:
        mod1, mod2 = result['feature_pair']
        metrics = result['avg_metrics']
        print(f"{mod1} vs {mod2}:")
        print(f"  R: {metrics['R_norm']:.4f}, U1: {metrics['U1_norm']:.4f}, "
              f"U2: {metrics['U2_norm']:.4f}, S: {metrics['S_norm']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WESAD Fine-Grained Modality RUS Analysis')
    parser.add_argument('--output_dir', type=str, default='../results/wesad',
                        help='Directory to save analysis results')
    parser.add_argument('--processed_dataset_dir', type=str, required=True,
                        help='Root directory of the processed WESAD datasets')
    parser.add_argument('--body_part', type=str, required=True,
                        help='Body part to analyze, either chest or wrist')
    parser.add_argument('--max_lag', type=int, default=100,
                        help='Number of lags to compute, evenly distributed across the sequence')
    parser.add_argument('--bins', type=int, default=8,
                        help='Number of bins for discretization (reduced for multivariate)')
    
    # General parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for batch method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    
    # BATCH method specific parameters
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches for batch method')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='Hidden dimension for neural networks in batch method')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers for neural networks in batch method')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'tanh'],
                        help='Activation function for neural networks in batch method')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for neural networks in batch method')
    parser.add_argument('--embed_dim', type=int, default=10,
                        help='Embedding dimension for alignment model in batch method')
    parser.add_argument('--discrim_epochs', type=int, default=40,
                        help='Number of epochs for discriminator training in batch method')
    parser.add_argument('--ce_epochs', type=int, default=10,
                        help='Number of epochs for CE alignment training in batch method')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)