# Temporal RUS for Time Series Classification

## Overview

This document describes the `temporal_rus_label.py` implementation, which adapts the temporal partial information decomposition (PID) approach for time series classification problems where the target variable Y is a discrete label rather than a continuous time series.

## Key Differences from `temporal_pid.py`

### Original Implementation (`temporal_pid.py`)
- **Input**: X1, X2, and Y are all time series (sequences)
- **Purpose**: Analyze temporal relationships between sequences
- **Distribution**: P(X1_past, X2_past, Y_present) with temporal lags

### New Implementation (`temporal_rus_label.py`)
- **Input**: X1 and X2 are univariate time series (1D arrays), Y is a scalar label
- **Purpose**: Analyze how temporal patterns in sequences relate to classification labels  
- **Distribution**: P(X1_features, X2_features, Y_label) from extracted temporal features

## Input Format

- **X1, X2**: Univariate time series with shape `(seq_length,)`
- **Y**: Scalar classification label (e.g., 0, 1, 2)

This aligns with the original `temporal_pid.py` approach but adapts it for classification tasks where we have a single label per time series pair.

## Key Components

### 1. Temporal Feature Extraction

The implementation provides three methods for extracting temporal features from sequences:

#### Sliding Window (`sliding_window`)
- Extracts overlapping windows of fixed size from the time series
- Captures local temporal patterns and dependencies
- Good for detecting specific temporal motifs

#### Summary Statistics (`summary_stats`)  
- Computes statistical measures (mean, std, min, max) over sliding windows
- Captures distributional properties of temporal segments
- More robust to noise and variations

#### First Differences (`differences`)
- Uses first differences to capture rate of change patterns
- Effective for detecting trends and oscillatory behaviors
- Emphasizes dynamic characteristics

### 2. Feature Aggregation

Multiple aggregation strategies are supported:
- **Max**: Takes maximum values across temporal windows
- **Mean**: Averages across temporal windows  
- **First/Last**: Uses first or last temporal window

### 3. PID Decomposition

The implementation decomposes mutual information I(Y; X1, X2) into:
- **Redundancy**: Information about Y shared by both X1 and X2 features
- **Unique X1**: Information about Y uniquely provided by X1 features
- **Unique X2**: Information about Y uniquely provided by X2 features  
- **Synergy**: Information about Y that emerges only from X1 and X2 together

## Synthetic Experiment Design

### Data Generation

The experiment generates 150 samples across 3 classes (50 samples each) with sequences of length 20:

**Class 0**: 
- X1: Increasing linear trend (0 to 2)
- X2: High-frequency sinusoidal oscillations (4π cycles)

**Class 1**:
- X1: Decreasing linear trend (2 to 0)  
- X2: Low-frequency sinusoidal oscillations (π cycles)

**Class 2** (Synergistic):
- X1: Step function (1 → 0 at midpoint)
- X2: Complementary step function (0 → 1 at midpoint)

### Experimental Results

The experiment tested all three feature extraction methods:

#### 1. Sliding Window Method
```
Redundancy:  0.1313 bits
Unique X1:   0.5992 bits  
Unique X2:   0.0000 bits
Synergy:     0.1477 bits
Total MI:    0.8781 bits
```
- **Interpretation**: X1 trends are well-captured by sliding windows
- Strong unique contribution from X1, limited X2 detection
- Some synergistic information present

#### 2. Summary Statistics Method
```
Redundancy:  0.2106 bits
Unique X1:   0.4274 bits
Unique X2:   0.4606 bits  
Synergy:     0.0000 bits
Total MI:    1.0986 bits
```
- **Interpretation**: Most balanced decomposition
- Both X1 and X2 contribute substantial unique information
- Higher redundancy, no synergy detected

#### 3. First Differences Method  
```
Redundancy:  0.1371 bits
Unique X1:   0.0054 bits
Unique X2:   0.2953 bits
Synergy:     0.1557 bits  
Total MI:    0.5935 bits
```
- **Interpretation**: X2 oscillations well-captured by differences
- Strong unique contribution from X2, limited X1 detection
- Some synergistic information present

### Key Insights

1. **Feature Method Sensitivity**: Different temporal feature extraction methods reveal different aspects of the information structure
2. **Complementary Information**: Summary statistics provide the most balanced view, while sliding windows and differences emphasize different temporal characteristics
3. **Perfect Decomposition**: All methods achieve perfect decomposition quality (Total MI = Sum of components)
4. **Synergy Detection**: The synergistic class (Class 2) is detected by sliding window and differences methods but not by summary statistics

## Generated Visualizations

The experiment produces four key visualizations:

1. **`synthetic_classification_data.png`**: Examples of the generated time series for each class
2. **`temporal_pid_classification_sliding_window.png`**: PID decomposition using sliding window features
3. **`temporal_pid_classification_summary_stats.png`**: PID decomposition using summary statistics features  
4. **`temporal_pid_classification_differences.png`**: PID decomposition using first differences features

## Applications

This approach is suitable for:
- Time series classification problems
- Understanding which temporal features are most informative
- Detecting synergistic patterns in multimodal time series
- Feature selection and extraction guidance for temporal data
- Causal analysis in classification contexts

## Technical Notes

- Uses convex optimization (CVXPY) for finding optimal Q distribution
- Handles discrete labels of arbitrary number of classes
- Supports both single and multiple time series inputs
- Includes robust error handling for optimization convergence
- All mutual information calculations converted to bits for interpretability

## Update: Temporal RUS Sequences

### Motivation

The original implementation only returned static RUS quantities (single values), which doesn't reveal how different temporal positions contribute to classification. The new temporal RUS sequence functionality addresses this by computing RUS quantities at different temporal lags, showing which parts of the time series are most informative.

### Implementation Details

For a single time series pair (X1, X2) with label Y:
1. **Sliding windows** are used to create multiple samples from the single sequence
2. Each window position at lag t provides temporal features
3. All windows share the same label Y
4. The distribution captures how temporal features relate to the classification label

### New Functions

#### `create_temporal_distribution_label()`
- Works with univariate time series (1D arrays)
- Creates probability distributions for specific temporal lags
- Uses sliding windows to generate samples from single sequences
- Binary encoding: features either predict label Y or not

#### `temporal_pid_label_sequence()`
- Computes RUS quantities for multiple temporal lags
- Returns arrays showing how information components change over time
- Reveals temporal dynamics of information flow

#### `plot_temporal_rus_sequences()`
- Visualizes temporal RUS sequences with four subplots:
  1. **Component Evolution**: Shows all RUS components vs lag
  2. **Stacked Components**: Area plot showing cumulative information
  3. **Normalized Components**: Percentage contribution at each lag
  4. **Unique Information Ratio**: X1/X2 dominance over time

### Example Results

From the synthetic experiment with 3 classes:

**Class 0 (Increasing trend + High frequency oscillations):**
- Peak total MI: 0.2433 bits at lag 27
- Peak redundancy: 0.0165 bits at lag 26
- Peak unique X2: 0.0231 bits at lag 30
- Peak synergy: 0.2187 bits at lag 27

**Class 1 (Decreasing trend + Low frequency oscillations):**
- Peak total MI: 0.2501 bits at lag 30
- Peak unique X1: 0.0235 bits at lag 28
- Peak synergy: 0.2215 bits at lag 25

**Class 2 (Synergistic step functions):**
- Peak redundancy: 0.0431 bits at lag 17
- Peak unique X1: 0.0426 bits at lag 23
- Substantial synergy throughout

### Key Insights from Temporal Analysis

1. **Different Classes Show Different Temporal Signatures**: Each class has distinct peaks in different RUS components at different lags

2. **Synergy Dominates**: For single sequence classification, synergy is often the largest component, indicating that X1 and X2 work together in complex ways

3. **Lag-Dependent Information**: The informative content varies significantly with temporal lag, suggesting that different parts of the sequence contribute differently to classification

### Applications of Temporal RUS Sequences

1. **Feature Engineering**: Identify which temporal lags to focus on
2. **Model Design**: Inform architecture choices (e.g., LSTM look-back windows)
3. **Interpretability**: Understand temporal decision-making in classifiers
4. **Data Collection**: Determine minimum sequence length needed
5. **Real-time Systems**: Know how much history is required for accurate classification

### Visualization Output

The temporal RUS sequence analysis produces a comprehensive 4-panel figure:
- **`temporal_rus_sequences_classification.png`**: Shows how information decomposition changes across temporal lags

This extension makes the tool particularly valuable for:
- Time series classification problems with varying temporal dependencies
- Understanding the temporal dynamics of multimodal sensor fusion
- Optimizing sliding window approaches in real-time systems
- Analyzing the temporal structure of sequential decision-making 