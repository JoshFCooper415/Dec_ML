# ML-Enhanced Production Planning

This project implements a machine learning-enhanced decomposition approach for multi-period, multi-scenario production planning problems.

## Overview

Production planning problems involve determining when to set up production and how much to produce in each time period to satisfy demand while minimizing costs. The ML-enhanced approach uses a neural network to accelerate the solution process by predicting good solutions for the subproblems.

## Project Structure

```
production_planning/
├── data/
│   ├── data_structures.py    # Common data structures
│   ├── data_loader.py        # Dataset for training
│   └── problem_generator.py  # Problem generation
├── models/
│   ├── neural_network.py     # Neural network architecture
│   ├── ml_predictor.py       # ML prediction interface
│   └── benders.py            # Benders decomposition solver
├── solvers/
│   ├── timeblock.py          # Time block subproblem
│   └── scenario.py           # Scenario subproblem
├── utils/
│   └── evaluation.py         # Evaluation utilities
├── standalone_train.py       # Training script with no dependencies
├── performance_evaluation.py # Performance evaluation script
├── data_generator.py         # Data generation for training
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Gurobi Optimizer
- NumPy, Pandas, Matplotlib
- tqdm (for progress bars)

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy pandas matplotlib tqdm gurobipy
   ```

## Quick Start Guide

### 1. Generate Training Data

Generate synthetic training data for the ML model:

```bash
python data_generator.py --output_dir data --block_sizes 3,6,9 --samples_per_size 500
```

This will create dataset files (`.npz`) for different block sizes.

### 2. Train Neural Network Models

Train the neural network models on the generated data:

```bash
python standalone_train.py --data_dir data --output_dir models --epochs 30
```

### 3. Evaluate Performance

Run a performance evaluation to measure the solution quality and computation time:

```bash
python performance_evaluation.py --problem_sizes 12:2,24:3,36:2 --trials 2 --output_dir results
```

This generates a comprehensive HTML report with performance metrics and visualizations.

## Command-Line Options

### Data Generation
- `--output_dir`: Output directory for data files
- `--block_sizes`: Comma-separated list of time block sizes
- `--samples_per_size`: Number of samples per block size

### Training
- `--data_dir`: Directory containing training data
- `--output_dir`: Output directory for models
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training

### Performance Evaluation
- `--problem_sizes`: Problem sizes as periods:scenarios (e.g., 12:2,24:3)
- `--seeds`: Random seeds for reproducibility
- `--trials`: Number of trials per problem size
- `--output_dir`: Output directory for results
- `--max_iterations`: Maximum Benders iterations

## Key Features

1. **Decomposition-Based Approach**: Uses Benders decomposition to split the problem into master and subproblems.

2. **Machine Learning Enhancement**: Neural networks predict solutions to subproblems, reducing computation time.

3. **Multi-Task Learning**: A single neural network predicts setup decisions, production quantities, and inventory levels.

4. **Performance Evaluation**: Comprehensive benchmarking comparing ML-enhanced vs. exact solutions.

## Performance Metrics

The performance evaluation script reports:
- Computation time
- Number of iterations
- MIP gap (optimality gap)
- Solution characteristics (setup decisions, inventory levels)

## Output Examples

The HTML report includes:
- Summary statistics tables
- Time vs. problem size plots
- MIP gap analysis
- Time breakdown charts

## Troubleshooting

- **Import errors**: Make sure your directory structure matches the project structure.
- **Gurobi license**: Ensure your Gurobi license is properly set up.
- **GPU usage**: If CUDA is available, the neural networks will use GPU acceleration.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
