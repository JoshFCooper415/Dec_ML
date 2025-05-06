"""
Record subproblem solutions for ML training.

This script uses the MLBendersDecomposition class to solve problems and records
the subproblem solutions to use as training data for the ML model.
"""

import os
import time
import pickle
import numpy as np
import random
from typing import List, Dict, Tuple, Set, Optional

# Import the required modules from the existing code
from data.problem_generator import create_test_problem
from data.data_structures import ScenarioData, ProblemParameters, TimingStats, Solution
from models.benders import MLBendersDecomposition
from solvers.scenario import MLScenarioSubproblem
from solvers.timeblock import MLTimeBlockSubproblem

def collect_subproblem_data(params: ProblemParameters, 
                           save_dir: str = "ml_training_data",
                           max_iterations: int = 20) -> Optional[Dict]:
    """
    Run Benders decomposition and record subproblem solutions for ML training.
    
    Args:
        params: Problem parameters
        save_dir: Directory to save training data
        max_iterations: Maximum number of Benders iterations
        
    Returns:
        Dictionary containing collected training data, or None if problem is infeasible
    """
    print(f"Collecting subproblem data for problem with {params.total_periods} periods and {len(params.scenarios)} scenarios...")
    
    # Create output directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Benders without ML
    benders = MLBendersDecomposition(params, use_ml=False)
    
    # Create containers for training data
    training_data = {
        'X_setup': [],      # Setup variables from master
        'X_params': [],     # Problem parameters
        'X_demands': [],    # Demands
        'X_init_inv': [],   # Initial inventory
        'y_setup': [],      # Optimal setup decisions
        'y_production': [], # Optimal production amounts
        'y_inventory': [],  # Optimal inventory levels
    }
    
    # Add a hook to capture subproblem solutions
    original_solve_recursive = MLScenarioSubproblem.solve_recursive
    
    def solve_recursive_with_capture(self, fixed_setups):
        """Wrapper to capture subproblem solutions during solve."""
        try:
            result, solution_data = original_solve_recursive(self, fixed_setups)
            
            # Only record feasible solutions
            if result != float('inf') and solution_data:
                # Process each time block
                for i, block in enumerate(self.time_blocks):
                    # Get block boundaries
                    start_idx = block.start_period
                    block_size = block.num_periods
                    
                    # Extract setup decisions from master
                    setup_features = np.zeros(block_size)
                    for t in range(block_size):
                        period = start_idx + t
                        if period in fixed_setups and fixed_setups[period]:
                            setup_features[t] = 1.0
                    
                    # Extract demands for this block
                    demands = np.array(self.scenario_data.demands[start_idx:start_idx+block_size])
                    
                    # Get problem parameters
                    params_vector = np.array([
                        self.params.capacity,
                        self.params.fixed_cost,
                        self.params.holding_cost,
                        self.params.production_cost
                    ])
                    
                    # Get initial inventory (from previous block or 0)
                    initial_inv = 0.0
                    if i > 0 and hasattr(block, 'model') and block.model:
                        try:
                            # Try to get initial inventory from the model variable
                            inv_var = block.model.getVarByName("inventory[0]")
                            if inv_var:
                                initial_inv = inv_var.x
                        except:
                            # Fall back to estimating from solution
                            initial_inv = solution_data['inventory'][start_idx-1] if start_idx > 0 else 0.0
                    
                    # Extract output variables (setups, production, inventory) for this block
                    y_setup = []
                    y_production = []
                    y_inventory = []
                    
                    for t in range(block_size):
                        period = start_idx + t
                        if period < len(solution_data['setup']):
                            y_setup.append(float(solution_data['setup'][period]))
                        else:
                            y_setup.append(0.0)
                            
                        if period < len(solution_data['production']):
                            y_production.append(solution_data['production'][period])
                        else:
                            y_production.append(0.0)
                            
                        if period < len(solution_data['inventory']):
                            y_inventory.append(solution_data['inventory'][period])
                        else:
                            y_inventory.append(0.0)
                    
                    # Add to training data
                    training_data['X_setup'].append(setup_features)
                    training_data['X_params'].append(params_vector)
                    training_data['X_demands'].append(demands)
                    training_data['X_init_inv'].append(initial_inv)
                    training_data['y_setup'].append(y_setup)
                    training_data['y_production'].append(y_production)
                    training_data['y_inventory'].append(y_inventory)
            
            return result, solution_data
        except Exception as e:
            print(f"Error in subproblem: {str(e)}")
            # Return infinity to indicate infeasibility
            return float('inf'), None
    
    # Monkey patch the method to capture data
    MLScenarioSubproblem.solve_recursive = solve_recursive_with_capture
    
    try:
        # Run Benders decomposition
        start_time = time.time()
        try:
            lower_bound, upper_bound, stats = benders.solve(max_iterations=max_iterations)
            
            # Check if the problem is infeasible
            if upper_bound == float('inf') or lower_bound == float('inf'):
                print("Problem is infeasible.")
                return None
                
            solve_time = time.time() - start_time
            
            print(f"Benders solved in {solve_time:.2f} seconds with {stats.num_iterations} iterations")
            print(f"Final bounds: LB={lower_bound:.2f}, UB={upper_bound:.2f}")
            print(f"Collected {len(training_data['X_setup'])} training samples")
            
            # Check if we actually collected any data
            if len(training_data['X_setup']) == 0:
                print("No training samples collected. Problem might be infeasible.")
                return None
            
            # Convert lists to numpy arrays
            for key in training_data:
                training_data[key] = np.array(training_data[key])
            
            # Save to file if a directory is provided
            if save_dir:
                block_size = training_data['X_setup'][0].shape[0] if len(training_data['X_setup']) > 0 else 0
                
                if block_size > 0:
                    output_file = os.path.join(save_dir, f"training_data_block{block_size}.npz")
                    np.savez(output_file, **training_data)
                    print(f"Training data saved to {output_file}")
            
            return training_data
            
        except Exception as e:
            print(f"Error during Benders solve: {str(e)}")
            return None
            
    finally:
        # Restore original method
        MLScenarioSubproblem.solve_recursive = original_solve_recursive

def generate_multiple_datasets(
    block_sizes: List[int] = [30],
    num_problems: int = 10,
    num_periods: int = 90,
    num_scenarios_range: Tuple[int, int] = (2, 5),
    c_values: List[int] = [3, 5, 8],  # C parameter values from the paper
    f_values: List[int] = [1000, 10000],  # F parameter values from the paper
    save_dir: str = "ml_training_data",
    max_retries: int = 5  # Maximum number of retries for infeasible problems
):
    """
    Generate multiple datasets for different block sizes, using specified C and F parameters.
    
    Args:
        block_sizes: List of block sizes to generate data for
        num_problems: Number of problems to generate
        num_periods: Number of periods per problem
        num_scenarios_range: Range of scenarios per problem
        c_values: List of capacity-to-demand ratio values (C parameter)
        f_values: List of setup-to-holding cost ratio values (F parameter)
        save_dir: Directory to save training data
        max_retries: Maximum number of retries for infeasible problems
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for block_size in block_sizes:
        print(f"\n===== Generating data for block size {block_size} =====")
        
        # Container for training data
        training_data = {
            'X_setup': [],
            'X_params': [],
            'X_demands': [],
            'X_init_inv': [],
            'y_setup': [],
            'y_production': [],
            'y_inventory': [],
            'problem_ids': [],
        }
        
        problem_counter = 0
        
        # Generate problems for each combination of C and F values
        for c_value in c_values:
            for f_value in f_values:
                for i in range(num_problems // (len(c_values) * len(f_values)) + 1):
                    if problem_counter >= num_problems:
                        break
                    
                    # Track number of retries for this problem
                    retry_count = 0
                    problem_data = None
                    
                    while problem_data is None and retry_count < max_retries:
                        # Randomly select number of scenarios
                        num_scenarios = random.randint(num_scenarios_range[0], num_scenarios_range[1])
                        
                        # Generate problem with specified parameters
                        seed = 42 + problem_counter + retry_count
                        
                        try:
                            params = create_test_problem(
                                num_periods=num_periods,
                                num_scenarios=num_scenarios,
                                periods_per_block=block_size,
                                capacity_to_demand_ratio=c_value,  # C parameter from paper
                                setup_to_holding_ratio=f_value,    # F parameter from paper
                                seed=seed
                            )
                            
                            print(f"Problem {problem_counter+1}/{num_problems}: {num_periods} periods, {num_scenarios} scenarios, C={c_value}, F={f_value}, Seed={seed}")
                            
                            # Collect data for this problem
                            problem_data = collect_subproblem_data(params, save_dir=None)
                            
                            if problem_data is None:
                                print(f"Problem with seed {seed} is infeasible. Retrying with new seed...")
                                retry_count += 1
                            else:
                                # Add problem ID to each sample
                                problem_ids = np.full(len(problem_data['X_setup']), seed)
                                
                                # Add to combined training data
                                for key in training_data:
                                    if key == 'problem_ids':
                                        training_data[key].extend(problem_ids)
                                    elif key in problem_data:
                                        training_data[key].extend(problem_data[key])
                                
                                print(f"Successfully collected data for problem with seed {seed}")
                                
                        except Exception as e:
                            print(f"Error generating or solving problem with seed {seed}: {str(e)}")
                            print(f"Retrying with new seed...")
                            retry_count += 1
                            problem_data = None
                    
                    if problem_data is None:
                        print(f"Failed to generate a feasible problem after {max_retries} attempts. Skipping...")
                    else:
                        problem_counter += 1
                    
                    # Break if we've collected enough problems
                    if problem_counter >= num_problems:
                        break
        
        # Check if we've collected any data
        if len(training_data['X_setup']) == 0:
            print(f"No training data collected for block size {block_size}. Skipping...")
            continue
            
        # Convert lists to numpy arrays
        for key in training_data:
            training_data[key] = np.array(training_data[key])
        
        # Save combined data
        output_file = os.path.join(save_dir, f"production_planning_ml_data_block{block_size}.npz")
        np.savez(output_file, **training_data)
        print(f"Combined training data saved to {output_file}")
        print(f"Total samples: {len(training_data['X_setup'])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training data from Benders decomposition')
    parser.add_argument('--block_sizes', type=str, default='3,6,9', help='Comma-separated list of block sizes')
    parser.add_argument('--num_problems', type=int, default=5, help='Number of problems per block size')
    parser.add_argument('--num_periods', type=int, default=90, help='Number of periods in each problem')
    parser.add_argument('--output_dir', type=str, default='ml_training_data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--c_values', type=str, default='3,5,8', help='Comma-separated list of C parameter values')
    parser.add_argument('--f_values', type=str, default='1000,10000', help='Comma-separated list of F parameter values')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum number of retries for infeasible problems')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse block sizes and parameter values
    block_sizes = [int(size) for size in args.block_sizes.split(',')]
    c_values = [int(c) for c in args.c_values.split(',')]
    f_values = [int(f) for f in args.f_values.split(',')]
    
    print(f"Generating data for block sizes: {block_sizes}")
    print(f"Number of problems per block size: {args.num_problems}")
    print(f"Number of periods: {args.num_periods}")
    print(f"C parameter values: {c_values}")
    print(f"F parameter values: {f_values}")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum retries: {args.max_retries}")
    
    # Generate datasets
    generate_multiple_datasets(
        block_sizes=block_sizes,
        num_problems=args.num_problems,
        num_periods=args.num_periods,
        c_values=c_values,
        f_values=f_values,
        save_dir=args.output_dir,
        max_retries=args.max_retries
    )