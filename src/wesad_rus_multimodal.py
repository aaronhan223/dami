import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
import pickle
import numpy as np
import itertools
import torch
from temporal_pid_multivariate import multi_lag_analysis

def load_wesad_data(subjects, processed_dataset_dir):
    subjects_data = []
    for subject in subjects:
        data_path = os.path.join(processed_dataset_dir, f'{subject}_processed.pkl')
        data = pickle.load(open(data_path, 'rb'))
        subjects_data.append(data)
    return subjects_data

def multi_lag_analysis_multi_subject(X1_list, X2_list, Y_list, max_lag, bins, **kwargs):
    results = {
        'lag': [],
        'redundancy': [],
        'unique_x1': [],
        'unique_x2': [],
        'synergy': [],
        'total_di': []
    }

    for lag in range(max_lag + 1):
        combined_X1_lag = []
        combined_X2_lag = []
        combined_Y_lag = []
        
        for X1_subj, X2_subj, Y_subj in zip(X1_list, X2_list, Y_list):
            if lag > 0:
                if len(Y_subj) <= lag:
                    continue
                X1_lagged = X1_subj[:-lag]
                X2_lagged = X2_subj[:-lag]
                Y_lagged = Y_subj[lag:]
            else:
                X1_lagged = X1_subj
                X2_lagged = X2_subj
                Y_lagged = Y_subj

            if len(Y_lagged) > 0:
                combined_X1_lag.append(X1_lagged)
                combined_X2_lag.append(X2_lagged)
                combined_Y_lag.append(Y_lagged)
        
        if not combined_X1_lag:
            continue
        
        X1_lag = np.vstack(combined_X1_lag)
        X2_lag = np.vstack(combined_X2_lag)
        Y_lag = np.concatenate(combined_Y_lag)

        lag_result = multi_lag_analysis(X1_lag, X2_lag, Y_lag, max_lag=0, bins=bins, **kwargs)
        results['lag'].append(lag)
        results['redundancy'].append(lag_result['redundancy'][0])
        results['unique_x1'].append(lag_result['unique_x1'][0])
        results['unique_x2'].append(lag_result['unique_x2'][0])
        results['synergy'].append(lag_result['synergy'][0])
        results['total_di'].append(lag_result['total_di'][0])

    return results
        



def main(args):
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    data_split = pickle.load(open(os.path.join(args.processed_dataset_dir, 'wesad_split.pkl'), 'rb'))
    train_subjects = data_split['train']
    
    
    train_subjects_data = load_wesad_data(train_subjects, args.processed_dataset_dir)

    modality_dim_dict = {'chest_signals': 8, 'wrist_signals': 6}
    modality_names = list(modality_dim_dict.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))

    dominant_pid_results = []
    all_pid_results = []

    for mod1, mod2 in modality_pairs:
        print(f"\n--- Analyzing Modality Pair {mod1} vs {mod2} ---")
        
        X1_list = [subject_data[mod1] for subject_data in train_subjects_data]
        X2_list = [subject_data[mod2] for subject_data in train_subjects_data]
        Y_list = [subject_data['labels'] for subject_data in train_subjects_data]

        pid_results = multi_lag_analysis_multi_subject(X1_list, X2_list, Y_list,
                                                       args.max_lag, args.bins,
                                                       method='batch',
                                                       batch_size=args.batch_size,
                                                       n_batches=args.n_batches,
                                                       seed=args.seed,
                                                       device=device,
                                                       hidden_dim=args.hidden_dim,
                                                       layers=args.layers,
                                                       activation=args.activation,
                                                       lr=args.lr,
                                                       embed_dim=args.embed_dim,
                                                       discrim_epochs=args.discrim_epochs,
                                                       ce_epochs=args.ce_epochs)
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
                'n_features_mod1': X1_list[0].shape[1],
                'n_features_mod2': X2_list[0].shape[1]
            })
        
    
    if all_pid_results:
        all_output_filename = f'rus_multimodal_all_lag{args.max_lag}.npy'
        all_output_path = os.path.join(args.output_dir, all_output_filename)
        print(f"Saving {len(all_pid_results)} PID results for all modality pairs to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True)
        print("Saving all modality pairs complete.")
    
    print(f"\nAnalysis complete for all {len(modality_pairs)} modality pairs.")


        
    

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIMIC-IV RUS computation')
    parser.add_argument('--output_dir', type=str, default='../results/wesad',
                        help='Directory to save analysis results')
    parser.add_argument('--processed_dataset_dir', type=str, required=True,
                        help='Root directory of the processed WESAD datasets')
    # parser.add_argument('--seq_len', type=int, default=100,
    #                     help='Length of sequences for lag computation. If None, inferred from data')
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Number of lags to compute, evenly distributed across the sequence')
    parser.add_argument('--bins', type=int, default=4,
                        help='Number of bins for discretization (reduced for multivariate)')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'joint', 'cvxpy', 'batch'],
                        help='PID estimation method (auto: choose based on dimensionality)')
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
    parser.add_argument('--hidden_dim', type=int, default=32,
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
    parser.add_argument('--discrim_epochs', type=int, default=100,
                        help='Number of epochs for discriminator training in batch method')
    parser.add_argument('--ce_epochs', type=int, default=10,
                        help='Number of epochs for CE alignment training in batch method')
    
    # CVXPY method specific parameters
    parser.add_argument('--regularization', type=float, default=1e-6,
                        help='Regularization parameter for CVXPY method')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)