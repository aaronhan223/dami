import os
import pickle
import numpy as np
import pandas as pd
from scipy import signal as scisig

# WESAD sampling frequencies
fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'Resp': 700}

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Apply butterworth lowpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    y = scisig.lfilter(b, a, data)
    return y

def downsample_to_1second(df):
    """
    Downsample aligned dataframe to 1-second resolution using appropriate aggregation methods
    
    Args:
        df: DataFrame with datetime index and aligned sensor data
    
    Returns:
        DataFrame with 1-second resolution
    """
    # Create 1-second time groups
    df_resampled = df.resample('1s')  # '1s' means 1 second intervals
    
    # Define aggregation methods for different types of data
    agg_dict = {}
    
    # For sensor columns: use multiple statistics to capture signal characteristics
    sensor_columns = [col for col in df.columns if col != 'label']
    for col in sensor_columns:
        agg_dict[col] = 'mean'
    
    # For labels: use mode (most frequent) to get the dominant label in each second
    if 'label' in df.columns:
        def get_mode_label(x):
            if x.empty:
                return np.nan
            mode_result = x.mode()
            return mode_result.iloc[0] if not mode_result.empty else x.iloc[0]
        agg_dict['label'] = get_mode_label
    
    # Apply aggregation
    df_1s = df_resampled.agg(agg_dict)
    
    # Drop any rows with NaN (shouldn't happen after our preprocessing, but just in case)
    df_1s = df_1s.dropna()
    
    return df_1s

def preprocess_wesad_subject(subject_path, target_fs=700):
    """
    Preprocess WESAD subject data to get aligned chest, wrist, and label arrays
    
    Args:
        subject_path: Path to subject .pkl file
        target_fs: Target sampling frequency (default: 700 Hz from labels)
    
    Returns:
        chest_signals: Array of shape (T, d1) where d1 is number of chest sensors
        wrist_signals: Array of shape (T, d2) where d2 is number of wrist sensors  
        labels: Array of shape (T,) with labels
    """
    
    # Load subject data
    with open(subject_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Get signals
    chest_data = data['signal']['chest']
    wrist_data = data['signal']['wrist']
    labels = data['label']
    print("len(labels)", len(labels))
    print("len chest_data acc", len(chest_data['ACC']))
    print("len chest_data ecg", len(chest_data['ECG']))
    print("len chest_data emg", len(chest_data['EMG']))
    print("len chest_data eda", len(chest_data['EDA']))
    print("len chest_data temp", len(chest_data['Temp']))
    print("len chest_data resp", len(chest_data['Resp']))
    print("Expected resulting length after alignment and downsampling", len(labels) // 700)
    # Create DataFrames with time indices for each signal type
    dataframes = {}
    
    # Process chest signals
    chest_signals = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
    for signal_name in chest_signals:
        if signal_name in chest_data:
            signal_data = chest_data[signal_name]
            
            # Handle multi-dimensional signals (like ACC)
            if signal_data.ndim > 1:
                for i in range(signal_data.shape[1]):
                    df = pd.DataFrame(signal_data[:, i], columns=[f'chest_{signal_name}_{i}'])
                    # Create time index based on sampling rate
                    if signal_name == 'ACC':
                        fs = fs_dict['ACC'] 
                    else:
                        fs = fs_dict.get(signal_name, 700)  # Default to 700 if not specified
                    df.index = [(1 / fs) * j for j in range(len(df))]
                    df.index = pd.to_datetime(df.index, unit='s')
                    dataframes[f'chest_{signal_name}_{i}'] = df
            else:
                df = pd.DataFrame(signal_data, columns=[f'chest_{signal_name}'])
                fs = fs_dict.get(signal_name, 700)
                df.index = [(1 / fs) * j for j in range(len(df))]
                df.index = pd.to_datetime(df.index, unit='s')
                dataframes[f'chest_{signal_name}'] = df
    
    # Process wrist signals
    wrist_signals = ['ACC', 'BVP', 'EDA', 'TEMP']
    for signal_name in wrist_signals:
        if signal_name in wrist_data:
            signal_data = wrist_data[signal_name]
            
            # Handle multi-dimensional signals (like ACC)
            if signal_data.ndim > 1:
                for i in range(signal_data.shape[1]):
                    df = pd.DataFrame(signal_data[:, i], columns=[f'wrist_{signal_name}_{i}'])
                    fs = fs_dict[signal_name]
                    df.index = [(1 / fs) * j for j in range(len(df))]
                    df.index = pd.to_datetime(df.index, unit='s')
                    dataframes[f'wrist_{signal_name}_{i}'] = df
            else:
                df = pd.DataFrame(signal_data, columns=[f'wrist_{signal_name}'])
                fs = fs_dict[signal_name]
                df.index = [(1 / fs) * j for j in range(len(df))]
                df.index = pd.to_datetime(df.index, unit='s')
                dataframes[f'wrist_{signal_name}'] = df
    
    # Process labels and create reference timeline
    label_df = pd.DataFrame(labels, columns=['label'])
    reference_timeline = [(1 / fs_dict['label']) * i for i in range(len(label_df))]
    reference_timeline = pd.to_datetime(reference_timeline, unit='s')
    label_df.index = reference_timeline
    
    # Start with labels as the reference DataFrame
    combined_df = label_df.copy()
    print(f"Started with reference timeline (labels): {len(combined_df)} rows")
    
    # For each sensor, interpolate to the reference timeline
    print("Aligning sensors to reference timeline:")
    for name, df in dataframes.items():
        print(f"  Interpolating {name} from {len(df)} to {len(combined_df)} samples")
        # Interpolate each sensor to the reference timeline
        for col in df.columns:
            # Use linear interpolation to map sensor data to reference timeline
            interpolated_values = np.interp(
                reference_timeline.astype('int64'),  # Reference timeline as integers
                df.index.astype('int64'),            # Original sensor timeline as integers
                df[col].values                       # Sensor values
            )
            combined_df[col] = interpolated_values
    
    # Labels are already properly aligned, no need for additional filling
    print(f"Final aligned DataFrame: {len(combined_df)} rows")
    
    # Drop any remaining NaN rows
    combined_df = combined_df.dropna()
    
    # Downsample to 1-second resolution for more manageable data size
    # Group by 1-second intervals and aggregate
    combined_df_1s = downsample_to_1second(combined_df)
    print("len combined_df_1s", len(combined_df_1s))
    # Separate chest, wrist, and label columns
    chest_cols = [col for col in combined_df_1s.columns if col.startswith('chest_')]
    wrist_cols = [col for col in combined_df_1s.columns if col.startswith('wrist_')]
    
    # Extract arrays
    chest_signals = combined_df_1s[chest_cols].values
    wrist_signals = combined_df_1s[wrist_cols].values
    labels = combined_df_1s['label'].values.astype(int)
    
    # Map labels 6 and 7 to 5 (unknown class)
    labels = np.where(labels == 6, 5, labels)
    labels = np.where(labels == 7, 5, labels)
    
    print(f"Label distribution after mapping: {np.bincount(labels)}")
    
    return chest_signals, wrist_signals, labels, chest_cols, wrist_cols

def preprocess_wesad_dataset(dataset_dir, subjects=None):
    """
    Preprocess multiple WESAD subjects
    
    Args:
        dataset_dir: Root directory containing subject folders
        subjects: List of subject IDs (e.g., ['S2', 'S3']). If None, processes all available
    
    Returns:
        Dictionary with subject data
    """
    
    if subjects is None:
        # Get all available subjects
        subjects = [d for d in os.listdir(dataset_dir) if d.startswith('S') and os.path.isdir(os.path.join(dataset_dir, d))]
        subjects.sort()
    
    processed_data = {}
    
    for subject in subjects:
        subject_path = os.path.join(dataset_dir, subject, f'{subject}.pkl')
        if os.path.exists(subject_path):
            print(f"Processing {subject}...")
            try:
                chest_signals, wrist_signals, labels, chest_cols, wrist_cols = preprocess_wesad_subject(subject_path)
                processed_data[subject] = {
                    'chest_signals': chest_signals,
                    'wrist_signals': wrist_signals, 
                    'labels': labels,
                    'chest_columns': chest_cols,
                    'wrist_columns': wrist_cols
                }
                print(f"  Chest shape: {chest_signals.shape}")
                print(f"  Wrist shape: {wrist_signals.shape}")
                print(f"  Labels shape: {labels.shape}")
            except Exception as e:
                print(f"  Error processing {subject}: {e}")
        else:
            print(f"  Subject file not found: {subject_path}")
    
    return processed_data

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WESAD dataset')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root directory of the WESAD dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed data')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Specific subjects to process (e.g., S2 S3)')
    
    args = parser.parse_args()
    
    # Process the data
    processed_data = preprocess_wesad_dataset(args.dataset_dir, args.subjects)
    
    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    
    for subject, data in processed_data.items():
        output_path = os.path.join(args.output_dir, f'{subject}_processed.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {subject} to {output_path}")
    
    # Also save all data together
    all_data_path = os.path.join(args.output_dir, 'all_subjects_processed.pkl')
    with open(all_data_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Saved all subjects to {all_data_path}")
