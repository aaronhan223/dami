import random
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description='Split WESAD dataset into train/validation/test sets')
parser.add_argument('--dataset_dir', type=str, required=True,
                    help='Root directory of the WESAD dataset')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Root directory of the WESAD dataset that will be used for storing the split subjects')
args = parser.parse_args()

# WESAD dataset subject IDs (excluding S1 and S12 which are not available)
SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
            'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

# Define split sizes for train/validation/test sets
train_num = 10
val_num = 2
test_num = 3

# Set random seed for reproducible splits
random.seed(42)
random.shuffle(SUBJECTS)

# Split subjects into train/val/test sets
train_subjects = SUBJECTS[:train_num]
val_subjects = SUBJECTS[train_num:train_num+val_num]
test_subjects = SUBJECTS[train_num+val_num:]

# Display the splits for verification
print(train_subjects)
print(val_subjects)
print(test_subjects)

# Save the subject splits to a pickle file for later use
os.makedirs(args.output_dir, exist_ok=True)
pickle.dump({'train': train_subjects, 'val': val_subjects, 'test': test_subjects}, open(os.path.join(args.output_dir, 'wesad_split.pkl'), 'wb'))