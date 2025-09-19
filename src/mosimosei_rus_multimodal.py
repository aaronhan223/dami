

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
import numpy as np
import itertools
import pickle
import torch
from multibench_affect_get_data import Affectdataset, drop_entry
from mimiciv_rus_multimodal import temporal_pid_label_multi_sequence_batch

def main(args):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    # Randomly sample num_subsample_stays from eligible stays
    np.random.seed(args.seed)

    dataset = pickle.load(open(args.dataset_path, 'rb'))
    processed_dataset = {'train': {}}
    processed_dataset['train'] = drop_entry(dataset['train'])
     
    train_dataset = Affectdataset(processed_dataset['train'], flatten_time_series=False, task='classification', max_pad=True, max_pad_num=50, data_type=args.dataset, z_norm=False)
    idx_dict = {
        'vision': 0,
        'audio': 1,
        'text': 2,
        'label': 3
    }
    if args.dataset == 'mosi':   
        modality_dim_dict = {
            'vision': 35,
            'audio': 74,
            'text': 300,
        }
    elif args.dataset == 'mosei':
        modality_dim_dict = {
            'vision': 713,
            'audio': 74,
            'text': 300,
        }
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    # train_dataloader, _, _ = get_dataloader(args.dataset_path, data_type=args.dataset, max_pad=True, max_seq_len=50, batch_size=args.batch_size)
    # train_dataloader = iter(train_dataloader)
    # batch = next(train_dataloader)
    # modality_dim_dict = {
    #     'vision': batch[0].shape[-1],
    #     'audio': batch[1].shape[-1],
    #     'text': batch[2].shape[-1],
    # }
    
    modality_names = list(modality_dim_dict.keys())
    modality_pairs = list(itertools.combinations(modality_names, 2))
    print(f"\nGenerated {len(modality_pairs)} pairs of modalities for analysis: {modality_pairs}")
    print(f"Modality dimensions: {modality_dim_dict}")
    dominant_pid_results = []
    all_pid_results = []

    for mod1, mod2 in modality_pairs:
        print(f"\n--- Analyzing Modality Pair {mod1} vs {mod2} ---")

        X1_list = [train_dataset[i][idx_dict[mod1]].numpy() for i in range(len(train_dataset))]
        X2_list = [train_dataset[i][idx_dict[mod2]].numpy() for i in range(len(train_dataset))]
        Y_list = [train_dataset[i][idx_dict['label']].numpy()[0,0] for i in range(len(train_dataset))]

        X1_masks = [np.ones(len(X), dtype=bool) for X in X1_list]
        X2_masks = [np.ones(len(X), dtype=bool) for X in X2_list]

        pid_results = temporal_pid_label_multi_sequence_batch(
            X1_list, X2_list, Y_list, X1_masks, X2_masks,
            lag=0, # always set lag to 0 because y is a static target
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            discrim_epochs=args.discrim_epochs,
            ce_epochs=args.ce_epochs,
            seed=args.seed,
            device=device,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            activation=args.activation,
            lr=args.lr,
            embed_dim=args.embed_dim,
            n_labels=len(np.unique(Y_list)),
            sequence_pooling=args.sequence_pooling
        )

        print(f"PID Results for {mod1} vs {mod2}: {pid_results}")


        r = pid_results['redundancy']
        u1 = pid_results['unique_x1']
        u2 = pid_results['unique_x2']
        s = pid_results['synergy']
        mi = pid_results['total_di']

        if mi > 1e-9:
            r_norm = r / mi
            u1_norm = u1 / mi
            u2_norm = u2 / mi
            s_norm = s / mi


            results = {
                'R_value': r,
                'U1_value': u1,
                'U2_value': u2,
                'S_value': s,
                'MI_value': mi,
                'R_norm': r_norm,
                'U1_norm': u1_norm,
                'U2_norm': u2_norm,
                'S_norm': s_norm
            }

            norm_values = {
                'R': r_norm,
                'U1': u1_norm,
                'U2': u2_norm,
                'S': s_norm
            }
            max_term = None
            max_value = -1
            for term, value in norm_values.items():
                if value > max_value:
                    max_value = value
                    max_term = term
            if max_term and max_value > args.dominance_threshold:
                print(f"Dominant term for {mod1} vs {mod2}: {max_term} with value {max_value}")
                dominant_pid_results.append({
                    'feature_pair': (mod1, mod2),
                    'dominant_term': max_term,
                    'dominance_ratio': 1, # always set to 1 because we only analyzed one lag
                    'lags_analyzed': 1,
                    'avg_metrics': results,
                    'lag_results': [results],
                    # 'modality1_features': modality_names[mod1],
                    # 'modality2_features': modality_names[mod2],
                    'n_features_mod1': X1_list[0].shape[1],
                    'n_features_mod2': X2_list[0].shape[1]
                })


            all_pid_results.append({
                'feature_pair': (mod1, mod2),
                'avg_metrics': results,
                'lag_results': [results],
                # 'modality1_features': modality_names[mod1],
                # 'modality2_features': modality_names[mod2],
                'n_features_mod1': X1_list[0].shape[1],
                'n_features_mod2': X2_list[0].shape[1]
            })

    if dominant_pid_results:
        output_filename = f'rus_multimodal_dominant_thresh{args.dominance_threshold:.1f}_{args.sequence_pooling}pool.npy'
        output_path = os.path.join(args.output_dir, args.dataset, output_filename)
        print(f"Saving {len(dominant_pid_results)} dominant multimodal PID results to {output_path}...")
        np.save(output_path, dominant_pid_results, allow_pickle=True)
        print("Saving dominant modality pairs complete.")
    
    if all_pid_results:
        all_output_filename = f'rus_multimodal_all_{args.sequence_pooling}pool.npy'
        all_output_path = os.path.join(args.output_dir, args.dataset, all_output_filename)
        print(f"Saving {len(all_pid_results)} PID results for all modality pairs to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True)
        print("Saving all modality pairs complete.")

    # --- Print summary of dominant terms ---
    if dominant_pid_results:
        dominance_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        for result in dominant_pid_results:
            term = result.get('dominant_term')
            if term in dominance_counts:
                dominance_counts[term] += 1

        print("\n--- Multimodal Dominance Summary ---")
        print(f"Total modality pairs with dominant terms: {len(dominant_pid_results)}")
        for term, count in dominance_counts.items():
            print(f"  {term} dominant: {count} pairs")
        print("----------------------------------------")

    else:
        print("No dominant PID terms found with the current threshold and percentage criteria.")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MOSI/MOSEI RUS computation')
    parser.add_argument('--output_dir', type=str, default='../results/affect',
                        help='Directory to save analysis results')
    parser.add_argument('--dataset', type=str, choices=['mosi', 'mosei'], required=True, help='Dataset to analyze, either MOSI or MOSEI')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the mosi_raw.pkl')
    parser.add_argument('--max_lag', type=int, default=10,
                        help='Maximum lag for temporal PID analysis')
    parser.add_argument('--bins', type=int, default=4,
                        help='Number of bins for discretization (reduced for multivariate)')
    parser.add_argument('--dominance_threshold', type=float, default=0.4,
                        help='Threshold for a PID term to be considered dominant')
    parser.add_argument('--dominance_percentage', type=float, default=0.9,
                        help='Percentage of lags a term must dominate to be considered dominant overall')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'joint', 'cvxpy', 'batch'],
                        help='PID estimation method (auto: choose based on dimensionality)')
    # General parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for batch method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    
    # BATCH method specific parameters
    parser.add_argument('--n_batches', type=int, default=10,
                        help='Number of batches for batch method')
    parser.add_argument('--hidden_dim', type=int, default=32,# 32,
                        help='Hidden dimension for neural networks in batch method')
    parser.add_argument('--layers', type=int, default=2, #2,
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
    parser.add_argument('--sequence_pooling', type=str, default='timestep',
                        choices=['timestep', 'mean'],
                        help='How to process sequences for batch method (timestep, mean)')
    # CVXPY method specific parameters
    parser.add_argument('--regularization', type=float, default=1e-6,
                        help='Regularization parameter for CVXPY method')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
    main(args)