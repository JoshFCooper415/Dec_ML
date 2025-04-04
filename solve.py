#!/usr/bin/env python
"""
Solve a production planning problem using ML-enhanced Benders decomposition.
"""

import argparse
import os

# Fixed imports to use the proper package structure
from data.problem_generator import create_test_problem
from models.benders import MLBendersDecomposition
from utils.evaluation import print_solution

def main():
    """Main entry point for solver script."""
    parser = argparse.ArgumentParser(description='Solve production planning problem')
    parser.add_argument('--periods', type=int, default=12, help='Number of time periods')
    parser.add_argument('--scenarios', type=int, default=2, help='Number of scenarios')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_dir', type=str, default='.', help='Directory containing model files')
    parser.add_argument('--use_ml', action='store_true', help='Use ML enhancement')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum number of iterations')
    
    args = parser.parse_args()
    
    # Create test problem
    print(f"Creating problem with {args.periods} periods, {args.scenarios} scenarios (seed {args.seed})...")
    params = create_test_problem(
        num_periods=args.periods,
        num_scenarios=args.scenarios,
        seed=args.seed
    )
    
    # Solve with or without ML
    if args.use_ml:
        print("\nSolving with ML-enhanced approach...")
        solver = MLBendersDecomposition(params, use_ml=True)
    else:
        print("\nSolving with exact approach (no ML)...")
        solver = MLBendersDecomposition(params, use_ml=False)
    
    try:
        lb, ub, stats = solver.solve(max_iterations=args.max_iterations)
        print(f"Solution found! Objective: {ub:.2f}")
        print(f"Solved in {stats.total_solve_time:.2f} seconds, {stats.num_iterations} iterations")
        
        if args.use_ml:
            print(f"ML prediction time: {stats.ml_prediction_time:.2f} seconds")
        
        # Print detailed solution
        print_solution(solver.best_solution, params)
        
    except Exception as e:
        print(f"Error solving problem: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()