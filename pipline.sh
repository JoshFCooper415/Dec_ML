#!/bin/bash
# Production Planning ML-Benders Pipeline
# This script:
# 1. Generates training data for different block sizes
# 2. Trains transformer models on the generated data
# 3. Runs benchmarks comparing direct solve vs ML-Benders approach

set -e  # Exit on error

# Configuration
DATA_DIR="./data"
MODEL_DIR="./models"
RESULTS_DIR="./results"
BLOCK_SIZES="12"  # Only use block size 12 for training
NUM_SAMPLES=2000     # Increase samples since we're only using one block size
TRAIN_EPOCHS=30      # Training epochs

# Problem sizes for benchmarking
PROBLEM_SIZES="12:2,24:3,36:4"  # Format: periods:scenarios
BENCHMARK_SEEDS="42,43,44"
BENCHMARK_TRIALS=3

# Create necessary directories
mkdir -p $DATA_DIR
mkdir -p $MODEL_DIR
mkdir -p $RESULTS_DIR

echo "==================================================="
echo "Production Planning ML-Benders Pipeline"
echo "==================================================="

# Step 1: Generate training data
echo "Step 1: Generating training data for block sizes: $BLOCK_SIZES"
python data_generator.py \
    --output_dir $DATA_DIR \
    --block_sizes $BLOCK_SIZES \
    --samples_per_size $NUM_SAMPLES \
    --seed 42 \
    --varied_params  # Use varied parameters for better generalization

echo "Data generation complete."

# Step 2: Train transformer models
echo "Step 2: Training transformer models on the generated data"
python train_model.py \
    --data_dir $DATA_DIR \
    --output_dir $MODEL_DIR \
    --epochs $TRAIN_EPOCHS \
    --batch_size 64 \
    --hidden_dim 256

echo "Model training complete."

# Step 3: Run benchmarks comparing approaches
echo "Step 3: Running benchmarks (Direct vs Benders vs ML-Benders)"

# First, run benchmark with all methods
python performance_evaluation.py \
    --problem_sizes $PROBLEM_SIZES \
    --seeds $BENCHMARK_SEEDS \
    --trials $BENCHMARK_TRIALS \
    --output_dir $RESULTS_DIR \
    --use_ml  # Enable ML (the script will run both with and without ML)