#!/usr/bin/env python3
"""
Benchmark script for comparing solution approaches for the production planning problem.
Run different solvers and compare their performance.
"""

import argparse
import sys
import os
import time
from typing import List, Tuple

# Import evaluation functions
from utils.evaluation import (
    run_benchmark, 
    plot_results,
    save_benchmark_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark different solvers for stochastic production planning")
    
    parser.add_argument("--problem-sizes", type=str, default="12,2;24,3;36,4",
                      help="Problem sizes to benchmark in format periods,scenarios;periods,scenarios")
    
    parser.add_argument("--seeds", type=str, default="42,43,44",
                      help="Random seeds to use for problem generation")
    
    parser.add_argument("--num-trials", type=int, default=3,
                      help="Number of trials to run for each problem size")
    
    parser.add_argument("--methods", type=str, default="direct,benders,ml_benders",
                      help="Methods to benchmark (comma-separated)")
    
    parser.add_argument("--timeout", type=int, default=600,
                      help="Timeout in seconds for each solver")
    
    parser.add_argument("--output", type=str, default="benchmark_results.pkl",
                      help="Output file for benchmark results")
    
    parser.add_argument("--plot", type=str, default="benchmark_plot.png",
                      help="Output file for benchmark plot")
    
    parser.add_argument("--no-plot", action="store_true",
                      help="Disable plotting")
    
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()


def parse_problem_sizes(sizes_str: str) -> List[Tuple[int, int]]:
    """Parse problem sizes from string.
    
    Args:
        sizes_str: Problem sizes in format "periods,scenarios;periods,scenarios"
        
    Returns:
        List of (periods, scenarios) tuples
    """
    problem_sizes = []
    for size_str in sizes_str.split(";"):
        periods, scenarios = map(int, size_str.split(","))
        problem_sizes.append((periods, scenarios))
    return problem_sizes


def main():
    """Main function to run benchmarks."""
    args = parse_args()
    
    # Parse problem sizes
    problem_sizes = parse_problem_sizes(args.problem_sizes)
    
    # Parse seeds
    seeds = list(map(int, args.seeds.split(",")))
    
    # Parse methods
    methods = args.methods.split(",")
    valid_methods = ['direct', 'benders', 'ml_benders']
    
    # Validate methods
    for method in methods:
        if method not in valid_methods:
            print(f"Error: Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")
            sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print benchmark configuration
    print("=" * 80)
    print("Benchmark Configuration:")
    print("=" * 80)
    print(f"Problem sizes: {problem_sizes}")
    print(f"Random seeds: {seeds}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Methods: {methods}")
    print(f"Timeout per solver: {args.timeout} seconds")
    print(f"Output file: {args.output}")
    if not args.no_plot:
        print(f"Plot file: {args.plot}")
    print("=" * 80)
    print()
    
    # Run benchmarks
    start_time = time.time()
    results = run_benchmark(
        problem_sizes=problem_sizes,
        seeds=seeds,
        num_trials=args.num_trials,
        methods=methods,
        timeout=args.timeout
    )
    total_time = time.time() - start_time
    
    # Save results
    save_benchmark_results(results, args.output)
    
    # Plot results
    if not args.no_plot:
        plot_results(results, args.plot)
    
    # Print summary
    print("\nBenchmark completed!")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Results saved to {args.output}")
    if not args.no_plot:
        print(f"Plots saved to {args.plot}")


if __name__ == "__main__":
    main()