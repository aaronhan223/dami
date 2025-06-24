import os
import pickle
import itertools
import pandas as pd
import numpy as np
import torch
from temporal_pid import multi_lag_analysis

DATASET_PATH = "/cis/home/xhan56/mimic-ts/XY_dl_data.pkl"
MIMICIV_DITEMS_PATH = "/cis/home/xhan56/mimic-ts/d_items.csv.gz"
OUTPUT_DIR = "../results/mimiciv"
MAX_LAG = 6
BINS = 8
DOMINANCE_THRESHOLD = 0.3 # Threshold for a PID term to be considered dominant

# item ids for meds of interest
SELECTED_MED_ITEMIDS = [223258, # insulin
                        228340, # furosemide (lasix)
                        # 225158, # 0.9% sodium chloride,
                        # 220949, # 5% dextrose in water (d5w)
                        # 225152, # heparin (continuous >= 4 days)
                        # 222168, # propofol infusion
                        # 225828, # lactated ringer's (lr)
                        ]

# item ids for chart of interest
SELECTED_CHART_ITEMIDS = [220621, # glucose (serum)
                          227442, # potassium (serum)
                          220635, # magnesium (serum)
                          225624, # BUN
                        #   220645, # sodium (serum)
                        #   220602, # chloride (serum)
                        #   220210, # respiratory rate
                        #   220277, # o2 saturation
                        #   227443, # bicarbonate (hco3)
                        #   227442, # potassium (serum)
                          ]

TARGET_ITEMID = 227442 # potassium (serum)


def get_selected_column_names():
    """Returns the human readable column names for chartevents and meds of interest"""
    ditems_df = pd.read_csv(MIMICIV_DITEMS_PATH, compression="gzip")
    itemid_to_label = dict(zip(ditems_df["itemid"], ditems_df["label"]))

    return [itemid_to_label[itemid] for itemid in SELECTED_MED_ITEMIDS], [itemid_to_label[itemid] for itemid in SELECTED_CHART_ITEMIDS]

def preprocess_mimciv_data(meds: torch.Tensor, chart: torch.Tensor, med_itemids: list, chart_itemids: list, num_patients: int = 100):
    """Select meds and chart of interest and filter out patients with sparse meds time series
    Args:
        meds: torch.Tensor, shape (n_samples, n_timesteps, n_features)
        chart: torch.Tensor, shape (n_samples, n_timesteps, n_features)
        med_itemids: list, shape (n_features,)
        chart_itemids: list, shape (n_features,)
        num_patients: int, number of patients to select
    Returns:
        meds: numpy.ndarray, shape (n_samples, n_timesteps, # of selected meds)
        chart: numpy.ndarray, shape (n_samples, n_timesteps, # of selected chart)
        target: numpy.ndarray, shape (n_samples, n_timesteps)
    """
    selected_med_column_idx_list = []
    for itemid in SELECTED_MED_ITEMIDS:
        selected_med_column_idx_list.append(med_itemids.index(str(itemid)))

    selected_chart_column_idx_list = []
    for itemid in SELECTED_CHART_ITEMIDS:
        selected_chart_column_idx_list.append(chart_itemids.index(str(itemid)))

    selected_target_column_idx = chart_itemids.index(str(TARGET_ITEMID))


    selected_meds = meds[:, :, selected_med_column_idx_list]
    selected_chart = chart[:, :, selected_chart_column_idx_list]
    selected_target = chart[:, :, selected_target_column_idx]


    med_nonzero_ratio = (selected_meds != 0).sum(dim=[1, 2]) / selected_meds.shape[1] / selected_meds.shape[2]

    sorted_med_nonzero_ratio_idx_list = torch.argsort(med_nonzero_ratio, descending=True)

    return selected_meds[sorted_med_nonzero_ratio_idx_list[:num_patients]].numpy(), \
           selected_chart[sorted_med_nonzero_ratio_idx_list[:num_patients]].numpy(), \
           selected_target[sorted_med_nonzero_ratio_idx_list[:num_patients]].numpy()

def main():
    """Main function to load, preprocess, and analyze MIMIC-IV data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading extracted MIMIC-IV data...")
    mimiciv = pickle.load(open(DATASET_PATH, "rb"))

    selected_meds_column_names, selected_chart_column_names = get_selected_column_names()
    print("Preprocessing extracted MIMIC-IV data...")
    selected_meds, selected_chart, selected_target = preprocess_mimciv_data(mimiciv["meds"], mimiciv["chart"], mimiciv["keys_to_cols"]["MEDS"], mimiciv["keys_to_cols"]["CHART"], num_patients=1)

    # squeeze to remove the batch dimension (there is only one patient)
    selected_meds, selected_chart, selected_target = selected_meds.squeeze(0), selected_chart.squeeze(0), selected_target.squeeze(0)

    # combine selected_meds and selected_chart
    selected_medschart = np.concatenate([selected_meds, selected_chart], axis=1)
    selected_medschart_column_names = selected_meds_column_names + selected_chart_column_names
    medschart_pairs = list(itertools.combinations(range(len(selected_medschart_column_names)), 2))

    dominant_pid_results = []
    all_pid_results = []

    for i, (idx1, idx2) in enumerate(medschart_pairs):
        col1 = selected_medschart_column_names[idx1]
        col2 = selected_medschart_column_names[idx2]
        print(f"Analyzing {col1} vs {col2}")
        X1 = selected_medschart[:, idx1]
        X2 = selected_medschart[:, idx2]
        Y = selected_target
        if len(X1) != len(Y) or len(X2) != len(Y):
            print(f"Warning: Length mismatch for pair ({col1}, {col2}). Skipping.")
            print(f"Len X1: {len(X1)}, Len X2: {len(X2)}, Len Y: {len(Y)}")
            continue

        print(f"Starting Temporal PID analysis...")
        print(f"X1: {col1} ({len(X1)} samples)")
        print(f"X2: {col2} ({len(X2)} samples)")
        print(f"Y: Potassium ({len(Y)} samples)")
        print(f"Max Lag: {MAX_LAG}, Bins: {BINS}")

        try:
            pid_results = multi_lag_analysis(X1, X2, Y, max_lag=MAX_LAG, bins=BINS)
        except Exception as e:
            print(f"Error during PID analysis for pair ({col1}, {col2}): {e}")
            continue

        # --- Store all PID results ---
        lags = pid_results.get('lag', range(MAX_LAG + 1)) # Assuming pid_results has 'lag' key or default to range
        for lag_idx, lag in enumerate(lags):
            try:
                r = pid_results['redundancy'][lag_idx]
                u1 = pid_results['unique_x1'][lag_idx] # Unique info from X1 (col1)
                u2 = pid_results['unique_x2'][lag_idx] # Unique info from X2 (col2)
                s = pid_results['synergy'][lag_idx]
                mi = pid_results['total_di'][lag_idx]

                # Store all results regardless of dominance
                all_pid_results.append({
                    'feature_pair': (col1, col2),
                    'lag': lag,
                    'R_value': r,
                    'U1_value': u1,
                    'U2_value': u2,
                    'S_value': s,
                    'MI_value': mi
                })

                # --- Check for dominance and collect dominant results for summary ---
                if mi > 1e-9: # Avoid division by zero or near-zero MI
                    dominant_term = None
                    if r / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'R'
                    elif u1 / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'U1'
                    elif u2 / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'U2'
                    elif s / mi > DOMINANCE_THRESHOLD:
                        dominant_term = 'S'

                    if dominant_term:
                        dominant_pid_results.append({
                            'feature_pair': (col1, col2),
                            'lag': lag,
                            'dominant_term': dominant_term,
                            'R_value': r,
                            'U1_value': u1,
                            'U2_value': u2,
                            'S_value': s,
                            'MI_value': mi
                        })
            except IndexError:
                 print(f"Warning: Index out of bounds for lag {lag} (index {lag_idx}) for pair ({col1}, {col2}). Skipping lag.")
                 continue
            except KeyError as e:
                 print(f"Warning: Missing key {e} in pid_results for pair ({col1}, {col2}). Skipping dominance check.")
                 break # Stop checking lags for this pair if keys are missing

    print(f"\nAnalysis complete for all {len(medschart_pairs)} pairs.")

    # --- Save all PID results ---
    if all_pid_results:
        all_output_filename = f'mimiciv_all_rus_lag{MAX_LAG}_bins{BINS}.npy'
        all_output_path = os.path.join(OUTPUT_DIR, all_output_filename)
        print(f"Saving {len(all_pid_results)} total PID results to {all_output_path}...")
        np.save(all_output_path, all_pid_results, allow_pickle=True) # Need allow_pickle=True for list of dicts
        print("Saving all results complete.")

    # --- Save dominant PID results ---
    if dominant_pid_results:
        output_filename = f'mimiciv_dominant_lag{MAX_LAG}_bins{BINS}_thresh{DOMINANCE_THRESHOLD:.1f}.npy'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        print(f"Saving {len(dominant_pid_results)} dominant PID results to {output_path}...")
        np.save(output_path, dominant_pid_results, allow_pickle=True) # Need allow_pickle=True for list of dicts
        print("Saving dominant results complete.")

        # --- Print summary of dominant terms ---
        dominance_counts = {'R': 0, 'U1': 0, 'U2': 0, 'S': 0}
        for result in dominant_pid_results:
            term = result.get('dominant_term')
            if term in dominance_counts:
                dominance_counts[term] += 1

        print("\n--- Dominance Summary ---")
        print(f"Total dominant instances found: {len(dominant_pid_results)}")
        for term, count in dominance_counts.items():
            print(f"  {term} dominant: {count} times")
        print("-------------------------")

    else:
        print("No dominant PID terms found with the current threshold.")

    print(f"Total RUS computations saved: {len(all_pid_results)}")

if __name__ == "__main__":
    main()
