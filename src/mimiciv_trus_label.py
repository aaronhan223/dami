import os
import pickle
import pandas as pd
import numpy as np
import torch
import itertools
from temporal_rus_label import temporal_pid_label_multi_sequence, plot_temporal_rus_sequences
import pdb

DATASET_PATH = "/cis/home/xhan56/mimic-ts/XY_dl_data.pkl"
MIMICIV_DITEMS_PATH = "/cis/home/xhan56/mimic-ts/d_items.csv"
OUTPUT_DIR = "../results/mimiciv"
MAX_LAG = 20
WINDOW_SIZE = 8
BINS = 20

# item ids for meds of interest
SELECTED_MED_ITEMIDS = [223258, # insulin
                        228340, # furosemide (lasix)
                        225936, # Replete with Fiber (Full), should not be predictive
                        225158,
                        220949,
                        225152,
                        222168,
                        225828
                       ]

# item ids for chart of interest
SELECTED_CHART_ITEMIDS = [225624, # BUN
                          223769, # O2 Saturation Pulseoxymetry Alarm - High
                          223761, # Temperature Fahrenheit, should not be predictive
                          220621,
                          227442,
                          220635,
                          220645,
                          220602,
                          220210,
                          220277,
                          227443
                         ]

def get_selected_column_names():
    """Returns the human readable column names for chartevents and meds of interest"""
    ditems_df = pd.read_csv(MIMICIV_DITEMS_PATH)
    itemid_to_label = dict(zip(ditems_df["itemid"], ditems_df["label"]))

    return [itemid_to_label[itemid] for itemid in SELECTED_MED_ITEMIDS], [itemid_to_label[itemid] for itemid in SELECTED_CHART_ITEMIDS]

def preprocess_mimiciv_data(meds: torch.Tensor, chart: torch.Tensor, targets: torch.Tensor, med_itemids: list, chart_itemids: list, num_patients: int = 1, return_dense_meds: bool = False):
    """Select meds and chart of interest and filter out patients with sparse meds time series
    Args:
        meds: torch.Tensor, shape (n_samples, n_timesteps, n_features)
        chart: torch.Tensor, shape (n_samples, n_timesteps, n_features)
        targets: torch.Tensor, shape (n_samples,)
        med_itemids: list, shape (n_features,)
        chart_itemids: list, shape (n_features,)
        num_patients: int, number of patients to select
    Returns:
        selected_meds: torch.Tensor, shape (n_samples, n_timesteps, n_features)
        selected_chart: torch.Tensor, shape (n_samples, n_timesteps, n_features)
        selected_target: torch.Tensor, shape (n_samples,)
    """
    selected_med_column_idx_list = []
    for itemid in SELECTED_MED_ITEMIDS:
        selected_med_column_idx_list.append(med_itemids.index(str(itemid)))

    selected_chart_column_idx_list = []
    for itemid in SELECTED_CHART_ITEMIDS:
        selected_chart_column_idx_list.append(chart_itemids.index(str(itemid)))

    selected_meds = meds[:, :, selected_med_column_idx_list]
    selected_chart = chart[:, :, selected_chart_column_idx_list]

    if return_dense_meds:
        med_nonzero_ratio = (selected_meds != 0).sum(dim=[1, 2]) / selected_meds.shape[1] / selected_meds.shape[2] # nonzero ratio for each patient (sparsity)
        
        sorted_med_nonzero_ratio_idx_list = torch.argsort(med_nonzero_ratio, descending=True)

        return selected_meds[sorted_med_nonzero_ratio_idx_list[:num_patients]].numpy(), \
            selected_chart[sorted_med_nonzero_ratio_idx_list[:num_patients]].numpy(), \
            targets[sorted_med_nonzero_ratio_idx_list[:num_patients]].numpy()
    else:
        # Randomly select num_patients indices
        total_patients = selected_meds.shape[0]
        random_indices = torch.randperm(total_patients)[:num_patients]
        
        return selected_meds[random_indices].numpy(), \
            selected_chart[random_indices].numpy(), \
            targets[random_indices].numpy()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading extracted MIMIC-IV data...")
    mimiciv = pickle.load(open(DATASET_PATH, "rb"))
    selected_meds_column_names, selected_chart_column_names = get_selected_column_names()
    print("Selected meds column names: ", selected_meds_column_names)
    print("Selected chart column names: ", selected_chart_column_names)

    selected_meds, selected_chart, selected_targets = preprocess_mimiciv_data(mimiciv["meds"], mimiciv["chart"], mimiciv["y"], mimiciv["keys_to_cols"]["MEDS"], mimiciv["keys_to_cols"]["CHART"], num_patients=40, return_dense_meds=True)
    print(f"Selected meds: {selected_meds.shape}")
    print(f"Selected chart: {selected_chart.shape}")
    print(f"Selected targets: {selected_targets}")
    # combine selected_meds and selected_chart
    selected_medschart = np.concatenate([selected_meds, selected_chart], axis=2)
    selected_medschart_column_names = selected_meds_column_names + selected_chart_column_names
    medschart_pairs = list(itertools.combinations(range(len(selected_medschart_column_names)), 2))
    print(f"Number of pairs: {len(medschart_pairs)}")
    all_results = {}

    for i, (idx1, idx2) in enumerate(medschart_pairs):
        col1 = selected_medschart_column_names[idx1]
        col2 = selected_medschart_column_names[idx2]
        print("************************************************")
        print(f"Analyzing {col1} vs {col2}")
        X1 = list(selected_medschart[:, :, idx1])
        X2 = list(selected_medschart[:, :, idx2])
        Y = list(selected_targets)
        
        temporal_results = temporal_pid_label_multi_sequence(X1, X2, Y, max_lag=MAX_LAG, bins=BINS)

        all_results[f"{col1}_{col2}"] = temporal_results
        print(f"\nTemporal analysis completed for lags 0 to {len(temporal_results['lags'] - 1)}")
        print(f"Peak total MI: {np.max(temporal_results['total_mi']):.4f} at lag {np.argmax(temporal_results['total_mi'])}")
        print(f"Peak redundancy: {np.max(temporal_results['redundancy']):.4f} at lag {np.argmax(temporal_results['redundancy'])}")
        print(f"Peak unique X1: {np.max(temporal_results['unique_x1']):.4f} at lag {np.argmax(temporal_results['unique_x1'])}")
        print(f"Peak unique X2: {np.max(temporal_results['unique_x2']):.4f} at lag {np.argmax(temporal_results['unique_x2'])}")
        print(f"Peak synergy: {np.max(temporal_results['synergy']):.4f} at lag {np.argmax(temporal_results['synergy'])}")
        
        plot_temporal_rus_sequences(
            temporal_results,
            title=f"Temporal RUS - X1: {col1}, X2: {col2}",
            save_path=os.path.join(OUTPUT_DIR, f'mimiciv_temporal_rus_{col1.replace("/", "-")}_{col2.replace("/", "-")}.png')
        )

if __name__ == "__main__":
    main()