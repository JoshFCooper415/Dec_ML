# ML-Enhanced Production Planning Quick Start Guide

This guide will help you get started with the ML-Enhanced Production Planning package.

## Prerequisites

- Python 3.8 or higher
- Gurobi Optimizer (with valid license)
- PyTorch

## Setup

1. Create and activate a virtual environment:

   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on Linux/Mac
   source venv/bin/activate
   ```

2. Install the package in development mode:

   ```bash
   pip install -e .
   ```

3. Install additional requirements:

   ```bash
   pip install tqdm
   ```

## Running the Complete Pipeline

The easiest way to get started is to run the complete pipeline, which will:

1. Generate synthetic training data
2. Train neural network models
3. Solve test problems
4. Run benchmarks

```bash
python run_pipeline.py
```

This will create three directories:
- `data/` - Contains generated training data
- `models/` - Contains trained neural network models
- `results/` - Contains benchmark results

## Running Individual Steps

### 1. Generating Training Data

Generate synthetic training data for different block sizes:

```bash
python -m production_planning.data_generator --output_dir data --block_sizes 3,6,9 --samples_per_size 1000
```

Options:
- `--output_dir` - Output directory for data files
- `--block_sizes` - Comma-separated list of time block sizes
- `--samples_per_size` - Number of samples per block size
- `--seed` - Random seed for reproducibility
- `--varied_params` - Vary problem parameters randomly

### 2. Training Models

Train neural network models on the generated data:

```bash
python -m production_planning.train --data_dir data --output_dir models
```

Options:
- `--data_dir` - Directory containing training data
- `--output_dir` - Output directory for models
- `--epochs` - Number of training epochs
- `--batch_size` - Batch size for training
- `--lr` - Learning rate
- `--hidden_dim` - Hidden dimension for neural network

### 3. Solving Problems

Solve a production planning problem:

```bash
# With ML enhancement
python -m production_planning.solve --periods 24 --scenarios 3 --use_ml

# Without ML (exact solution)
python -m production_planning.solve --periods 24 --scenarios 3
```

Options:
- `--periods` - Number of time periods
- `--scenarios` - Number of scenarios
- `--seed` - Random seed
- `--model_dir` - Directory containing models
- `--use_ml` - Enable ML enhancement
- `--max_iterations` - Maximum number of Benders iterations

### 4. Running Benchmarks

Run benchmarks to compare ML-enhanced and exact solutions:

```bash
python -m production_planning.benchmark --output results/benchmark_results.pkl
```

Options:
- `--output` - Output file for benchmark results
- `--model_dir` - Directory containing models
- `--trials` - Number of trials per problem size
- `--max_size` - Maximum problem size index to test

## Visualizing Results

To visualize the benchmark results, you can use the following code:

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results
with open("results/benchmark_results.pkl", "rb") as f:
    results = pickle.load(f)

# Filter successful results
success_results = [r for r in results if r["status"] == "success"]

# Extract data for plotting
periods = [r["periods"] for r in success_results]
speedups = [r["speedup"] for r in success_results]
obj_diffs = [r["obj_rel_diff"] for r in success_results]

# Plot speedup vs problem size
plt.figure(figsize=(10, 6))
plt.scatter(periods, speedups)
plt.xlabel("Number of Periods")
plt.ylabel("Speedup (×)")
plt.title("ML Speedup vs Problem Size")
plt.grid(True)
plt.savefig("results/speedup_plot.png")

# Plot optimality gap vs problem size
plt.figure(figsize=(10, 6))
plt.scatter(periods, obj_diffs)
plt.xlabel("Number of Periods")
plt.ylabel("Optimality Gap (%)")
plt.title("ML Optimality Gap vs Problem Size")
plt.grid(True)
plt.savefig("results/gap_plot.png")

print("Results visualized and saved to results/ directory")
```

## Troubleshooting

If you encounter any issues:

1. **Gurobi License Error**: Make sure your Gurobi license is properly set up
2. **Memory Error**: Reduce batch size or number of samples
3. **Import Error**: Make sure the package is installed in development mode with `pip install -e .`
4. **CUDA Error**: Set device to CPU if GPU is not available or has limited memory