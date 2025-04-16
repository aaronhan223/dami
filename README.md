# Directed Information-Based Causal Discovery

This project implements a method for discovering causal structures in multivariate time series data using directed information. It specifically addresses the challenge of identifying causal relationships with different time lags between variables.

## Features

- **Time-Lagged Causal Discovery**: Discovers causal relationships with different time lags between variables
- **Directed Information Computation**: Implements directed information calculations to quantify causal influence
- **Optimal Lag Identification**: Automatically identifies the optimal time lag for each causal relationship
- **Synthetic Data Generation**: Includes functionality to generate synthetic data with known causal structures
- **Performance Evaluation**: Provides tools to evaluate the accuracy of the discovered causal structures
- **Multiple Test Cases**: Tests the algorithm on various causal structures with different complexity levels

## Requirements

The script requires the following Python packages:
- numpy
- matplotlib
- networkx
- scipy
- scikit-learn
- pandas
- seaborn
- tqdm

You can install them using:
```
pip install -r requirements.txt
```

## Usage

To run the causal discovery algorithm on the predefined test cases:

```python
python directed_information_causal_discovery.py
```

This will:
1. Generate synthetic time series data for multiple causal structures
2. Apply the causal discovery algorithm to each dataset
3. Evaluate the performance of the algorithm
4. Generate visualizations of the true and discovered causal graphs
5. Output performance metrics for each test case

## How It Works

The algorithm follows these key steps:

1. **Temporal Data Generation**: Creates synthetic multivariate time series with specified causal relationships and time lags.

2. **Time Lag Discovery**: For each pair of variables, it identifies the optimal time lag that maximizes the directed information.

3. **Directed Information Calculation**: Calculates the directed information from one variable to another at the identified optimal lag.

4. **Graph Construction**: Constructs a causal graph where an edge is added if the directed information exceeds a specified threshold.

5. **Performance Evaluation**: Compares the discovered causal graph with the ground truth to evaluate the algorithm's performance.

## Output

The script generates:
- Visualizations of the true and discovered causal graphs for each test case
- A summary of performance metrics across all test cases
- A bar plot comparing the performance across different causal structures

## Mathematical Foundation

Directed information is defined as:
I(X → Y) = ∑ I(X^n → Y_n | Y^{n-1})

Where:
- X^n represents the past of variable X up to time n
- Y_n represents variable Y at time n
- Y^{n-1} represents the past of variable Y up to time n-1
- I(X → Y | Z) is the conditional mutual information

## License

MIT License
