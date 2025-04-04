#!/usr/bin/env python
"""
Benchmark ML-enhanced Benders decomposition against exact solution.
"""

import argparse
import os

# Fixed imports to use the proper package structure
from utils.evaluation import run_benchmark, save_benchmark_results

def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description='Benchmark ML-enhanced production planning')
    parser.add_argument('--output', type=str, default='benchmark_results.pkl', help='Output file for results')
    parser.add_argument('--model_dir', type=str, default='.', help='Directory containing model files')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per problem size')
    parser.add_argument('--max_size', type=int, default=3, help='Maximum problem size index to test')
    
    args = parser.parse_args()
    
    # Define problem sizes to benchmark
    problem_sizes = [
        (12, 2),  # (periods, scenarios)
        (18, 3),
        (24, 4),
        (36, 5),
        (48, 6)
    ]
    
    # Use only up to max_size
    problem_sizes = problem_sizes[:args.max_size]
    
    # Run benchmarks
    print("Running ML-enhanced Benders decomposition benchmarks...")
    
    results = run_benchmark(
        problem_sizes=problem_sizes,
        num_trials=args.trials
    )
    
    # Save results
    save_benchmark_results(results, args.output)
    
    return 0

if __name__ == "__main__":
    main()