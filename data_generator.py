"""
Standalone data generator for production planning ML.

This script generates training data without depending on any external modules.
"""

import os
import argparse
import numpy as np
import random
import time
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Using simple progress reporting.")
    # Create a simple tqdm replacement
    class tqdm:
        def __init__(self, total, desc=None):
            self.total = total
            self.n = 0
            self.desc = desc
            if desc:
                print(f"{desc}: 0/{total}")
            
        def update(self, n):
            self.n += n
            if self.desc:
                print(f"{self.desc}: {self.n}/{self.total}")
            
        def close(self):
            pass

# Define data structures inline to avoid import issues
@dataclass
class ScenarioData:
    """Data class for storing scenario information."""
    demands: List[float]
    probability: float

@dataclass
class ProblemParameters:
    """Data class for storing problem parameters."""
    total_periods: int
    periods_per_block: int
    capacity: float
    fixed_cost: float
    holding_cost: float
    production_cost: float
    scenarios: List[ScenarioData]
    linking_periods: Set[int]

def create_test_problem(num_periods: int = 12, num_scenarios: int = 3, seed: int = 42) -> ProblemParameters:
    """Create a test problem instance."""
    random.seed(seed)
    
    scenarios = []
    
    # Generate random demand scenarios
    base_demand = 100
    for _ in range(num_scenarios):
        demands = [
            base_demand * (1 + 0.3 * random.uniform(-1, 1))
            for _ in range(num_periods)
        ]
        scenarios.append(ScenarioData(
            demands=demands,
            probability=1.0/num_scenarios
        ))
    
    # Define linking periods (every third period)
    linking_periods = set(range(0, num_periods, 3))
    
    return ProblemParameters(
        total_periods=num_periods,
        periods_per_block=3,
        capacity=300,
        fixed_cost=1000,
        holding_cost=5,
        production_cost=10,
        scenarios=scenarios,
        linking_periods=linking_periods
    )

def solve_single_block(params, start_period, num_periods, initial_inventory, scenario_data):
    """Solve a single time block subproblem."""
    model = gp.Model("TimeBlock")
    model.setParam('OutputFlag', 0)
    
    # Variables
    setup_vars = {}
    prod_vars = {}
    inv_vars = {}
    
    for t in range(num_periods):
        setup_vars[t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{t}]")
        prod_vars[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"production[{t}]")
    
    for t in range(num_periods + 1):
        inv_vars[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"inventory[{t}]")
    
    # Fix initial inventory
    inv_vars[0].lb = initial_inventory
    inv_vars[0].ub = initial_inventory
    
    # Constraints
    for t in range(num_periods):
        period = start_period + t
        
        # Inventory balance
        model.addConstr(
            inv_vars[t] + prod_vars[t] == 
            scenario_data.demands[period] + inv_vars[t+1],
            name=f"balance_{t}"
        )
        
        # Capacity constraint
        model.addConstr(
            prod_vars[t] <= params.capacity * setup_vars[t],
            name=f"capacity_{t}"
        )
    
    # Objective
    obj = (gp.quicksum(params.fixed_cost * setup_vars[t] + 
                      params.production_cost * prod_vars[t] + 
                      params.holding_cost * inv_vars[t]
                      for t in range(num_periods)))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise ValueError("Failed to solve time block subproblem")
    
    # Extract solution
    setup = [setup_vars[t].x > 0.5 for t in range(num_periods)]
    production = [prod_vars[t].x for t in range(num_periods)]
    inventory = [inv_vars[t].x for t in range(num_periods)]
    
    return {
        'setup': setup,
        'production': production,
        'inventory': inventory
    }

def generate_training_data(block_size, num_samples, seed=42, varied_params=True):
    """Generate training data for a specific block size."""
    random.seed(seed)
    np.random.seed(seed)
    
    X_setup = []
    X_params = []
    X_demands = []
    X_init_inv = []
    
    y_setup = []
    y_production = []
    y_inventory = []
    
    problem_ids = []
    sample_idx = 0
    
    # Progress tracking
    pbar = tqdm(total=num_samples, desc=f"Generating data for block size {block_size}")
    
    while sample_idx < num_samples:
        # Randomly vary parameters if requested
        if varied_params:
            capacity = random.uniform(200, 400)
            fixed_cost = random.uniform(800, 1200)
            holding_cost = random.uniform(3, 7)
            production_cost = random.uniform(8, 12)
            num_periods = random.randint(max(block_size + 3, 12), max(block_size + 10, 24))
            num_scenarios = 1  # We generate one scenario at a time for training data
        else:
            capacity = 300
            fixed_cost = 1000
            holding_cost = 5
            production_cost = 10
            num_periods = max(block_size + 6, 16)
            num_scenarios = 1
        
        # Create a problem instance
        problem_seed = seed + sample_idx
        params = create_test_problem(
            num_periods=num_periods,
            num_scenarios=num_scenarios,
            seed=problem_seed
        )
        
        # Override with our parameters
        params.capacity = capacity
        params.fixed_cost = fixed_cost
        params.holding_cost = holding_cost
        params.production_cost = production_cost
        params.periods_per_block = block_size
        
        # Randomly sample some initial inventory values
        max_blocks = (num_periods - block_size + 1)
        num_blocks_to_sample = min(max_blocks, 3)  # Sample up to 3 blocks per problem
        
        for _ in range(num_blocks_to_sample):
            # Random start period (ensure we have a full block)
            start_period = random.randint(0, num_periods - block_size)
            
            # Random initial inventory
            initial_inv = random.uniform(0, capacity * 0.5)
            
            # Random fixed setups for some periods
            fixed_setups = {}
            setup_features = np.zeros(block_size)
            
            # With 30% probability, fix some setups
            if random.random() < 0.3:
                num_fixed = random.randint(1, max(1, block_size // 3))
                fixed_periods = random.sample(range(block_size), num_fixed)
                
                for t in fixed_periods:
                    is_setup = random.random() > 0.5
                    fixed_setups[start_period + t] = is_setup
                    setup_features[t] = 1 if is_setup else 0
            
            try:
                # Solve the block
                solution = solve_single_block(
                    params=params,
                    start_period=start_period,
                    num_periods=block_size,
                    initial_inventory=initial_inv,
                    scenario_data=params.scenarios[0]  # First scenario
                )
                
                # Extract demands for this block
                demands = params.scenarios[0].demands[start_period:start_period + block_size]
                
                # Record features
                X_setup.append(setup_features)
                X_params.append([capacity, fixed_cost, holding_cost, production_cost])
                X_demands.append(demands)
                X_init_inv.append(initial_inv)
                
                # Record targets
                y_setup.append(solution['setup'])
                y_production.append(solution['production'])
                y_inventory.append(solution['inventory'])
                
                # Track problem ID
                problem_ids.append(problem_seed)
                
                sample_idx += 1
                pbar.update(1)
                
                # Break if we've collected enough samples
                if sample_idx >= num_samples:
                    break
                    
            except Exception as e:
                print(f"Error solving problem: {str(e)}")
                continue
    
    pbar.close()
    
    # Convert to numpy arrays
    X_setup = np.array(X_setup, dtype=float)
    X_params = np.array(X_params, dtype=float)
    X_demands = np.array(X_demands, dtype=float)
    X_init_inv = np.array(X_init_inv, dtype=float)
    
    y_setup = np.array(y_setup, dtype=float)
    y_production = np.array(y_production, dtype=float)
    y_inventory = np.array(y_inventory, dtype=float)
    
    problem_ids = np.array(problem_ids, dtype=int)
    
    # Return data in dictionary format
    return {
        'X_setup': X_setup,
        'X_params': X_params,
        'X_demands': X_demands,
        'X_init_inv': X_init_inv,
        'y_setup': y_setup,
        'y_production': y_production,
        'y_inventory': y_inventory,
        'problem_ids': problem_ids
    }

def main():
    """Main entry point for data generation script."""
    parser = argparse.ArgumentParser(description='Generate training data for production planning')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--block_sizes', type=str, default='3,6,9', help='Comma-separated list of block sizes')
    parser.add_argument('--samples_per_size', type=int, default=1000, help='Number of samples per block size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--varied_params', action='store_true', help='Vary problem parameters')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse block sizes
    block_sizes = [int(size) for size in args.block_sizes.split(',')]
    
    for block_size in block_sizes:
        print(f"Generating data for block size {block_size}...")
        start_time = time.time()
        
        data = generate_training_data(
            block_size=block_size,
            num_samples=args.samples_per_size,
            seed=args.seed,
            varied_params=args.varied_params
        )
        
        # Save to file
        output_file = os.path.join(args.output_dir, f"production_planning_ml_data_block{block_size}.npz")
        np.savez(output_file, **data)
        
        duration = time.time() - start_time
        print(f"Generated {args.samples_per_size} samples for block size {block_size} in {duration:.2f} seconds")
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()