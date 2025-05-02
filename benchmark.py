#!/usr/bin/env python3
"""
Enhanced benchmark script for comparing solution approaches for stochastic production planning.
This script:
1. Runs different solvers (direct, benders, ml_benders) on various problem sizes
2. Tests combinations of C (capacity-to-demand ratio) and F (setup-to-holding cost ratio) values
3. Collects and reports performance metrics
4. Generates comprehensive visualizations and HTML reports
5. Enforces solver timeout limit
"""
from data.problem_generator import create_test_problem
from models.benders import MLBendersDecomposition
from solvers.direct_solver import DirectSolver
import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple, Dict, Any
import datetime
from tqdm import tqdm
import signal
import threading

# Define a TimeoutException
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Solver exceeded time limit")

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=600):
    """Run a function with a timeout.
    
    Args:
        func: The function to run
        args: The positional arguments to pass to the function
        kwargs: The keyword arguments to pass to the function
        timeout_duration: The timeout duration in seconds
        
    Returns:
        The result of the function if it completes within the timeout,
        or raises a TimeoutException otherwise.
    """
    # Use a different approach on different platforms
    import platform
    
    result = [None]
    exception = [None]
    
    if platform.system() == 'Windows':
        # Windows doesn't support SIGALRM, so use a threading approach
        def worker():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout_duration)
        
        if thread.is_alive():
            raise TimeoutException("Solver exceeded time limit")
        
        if exception[0] is not None:
            raise exception[0]
        
        return result[0]
    else:
        # Unix-based systems can use SIGALRM
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)
        
        try:
            result[0] = func(*args, **kwargs)
        except TimeoutException:
            raise
        except Exception as e:
            exception[0] = e
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        if exception[0] is not None:
            raise exception[0]
        
        return result[0]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced benchmark for stochastic production planning solvers")
    
    parser.add_argument("--problem-sizes", type=str, default="12,2;24,3;36,4",
                      help="Problem sizes to benchmark in format periods,scenarios;periods,scenarios")
    
    parser.add_argument("--seeds", type=str, default="42,43,44",
                      help="Random seeds to use for problem generation")
    
    parser.add_argument("--num-trials", type=int, default=3,
                      help="Number of trials to run for each problem size")
    
    parser.add_argument("--methods", type=str, default="direct,benders,ml_benders",
                      help="Methods to benchmark (comma-separated)")
    
    parser.add_argument("--c-values", type=str, default="3,5,8",
                      help="Capacity-to-demand ratios to test (comma-separated)")
    
    parser.add_argument("--f-values", type=str, default="1000,10000",
                      help="Setup-to-holding cost ratios to test (comma-separated)")
    
    parser.add_argument("--timeout", type=int, default=600,
                      help="Timeout in seconds for each solver")
    
    parser.add_argument("--max-iterations", type=int, default=50,
                      help="Maximum number of Benders iterations/subproblems")
    
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Output directory for benchmark results")
    
    parser.add_argument("--report-name", type=str, default=None,
                      help="Custom name for the report (default: auto-generated)")
    
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


def run_benchmark(
    problem_sizes: List[Tuple[int, int]],
    seeds: List[int],
    c_values: List[int],
    f_values: List[int],
    num_trials: int,
    methods: List[str],
    timeout: int,
    max_iterations: int = 20_000,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Run benchmarks on different problem sizes and methods.
    
    Args:
        problem_sizes: List of (periods, scenarios) tuples
        seeds: List of random seeds
        c_values: List of capacity-to-demand ratio values to test
        f_values: List of setup-to-holding cost ratio values to test
        num_trials: Number of trials to run for each configuration
        methods: List of methods to benchmark
        timeout: Timeout in seconds for each solver
        max_iterations: Maximum Benders iterations/subproblems
        verbose: Whether to print detailed progress
        
    Returns:
        List of result dictionaries
    """
    # Dictionary to store direct solve results for OPT gap calculation
    direct_results = {}
    
    # Total number of runs
    total_runs = len(problem_sizes) * len(c_values) * len(f_values) * min(num_trials, len(seeds)) * len(methods)
    run_count = 0
    
    # Results list
    results = []
    
    # For each problem size
    for periods, scenarios in problem_sizes:
        # For each C value
        for c_value in c_values:
            # For each F value
            for f_value in f_values:
                # For each random seed (up to num_trials)
                for seed in seeds[:num_trials]:
                    
                    # Create problem
                    print(f"\nGenerating problem with {periods} periods, {scenarios} scenarios, C={c_value}, F={f_value} (seed {seed})...")
                    
                    # Create a real test problem
                    try:
                        params = create_test_problem(
                            num_periods=periods, 
                            num_scenarios=scenarios, 
                            capacity_to_demand_ratio=c_value,
                            setup_to_holding_ratio=f_value,
                            seed=seed
                        )
                        
                        # Record problem characteristics
                        problem_info = {
                            'periods': periods,
                            'scenarios': scenarios,
                            'c_value': c_value,
                            'f_value': f_value,
                            'seed': seed,
                            'problem_size': periods * scenarios,
                            'total_demand': sum(sum(s.demands) for s in params.scenarios) / len(params.scenarios),
                            'demand_variability': np.std([item for s in params.scenarios for item in s.demands])
                        }
                    except Exception as e:
                        print(f"Error creating problem: {str(e)}")
                        continue
                    
                    # For each solution method
                    for method in methods:
                        print(f"Solving with method: {method}")
                        run_count += 1
                        print(f"Run {run_count}/{total_runs}")
                        
                        # Create the appropriate solver
                        try:
                            start_time = time.time()
                            
                            if method == "direct":
                                # Use the real DirectSolver
                                solver = DirectSolver(params)
                                
                                try:
                                    # Run with timeout
                                    objective, stats = run_with_timeout(
                                        solver.solve, 
                                        timeout_duration=timeout
                                    )
                                    
                                    lower_bound = objective  # Direct solver achieves optimality
                                    subproblems = 1
                                    master_time = stats.master_time if hasattr(stats, 'master_time') else 0
                                    subproblem_time = stats.subproblem_time if hasattr(stats, 'subproblem_time') else 0
                                    mip_gap = 0.0  # Direct solver achieves optimality
                                    
                                    # Extract solution details
                                    solution = solver.best_solution
                                    
                                except TimeoutException:
                                    print(f"  Timeout occurred after {timeout} seconds.")
                                    
                                    # Create a result indicating timeout
                                    result = {
                                        **problem_info,
                                        'method': method,
                                        'status': 'timeout',
                                        'total_time': timeout,
                                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    results.append(result)
                                    continue
                                
                            elif method == "benders":
                                # Use the real Benders Decomposition without ML
                                solver = MLBendersDecomposition(
                                    params, 
                                    use_ml=False,
                                    use_trust_region=False,
                                    use_laporte_cuts=False
                                )
                                
                                try:
                                    # Run with timeout
                                    lower_bound, objective, stats = run_with_timeout(
                                        solver.solve, 
                                        kwargs={'max_iterations': max_iterations},
                                        timeout_duration=timeout
                                    )
                                    
                                    subproblems = stats.num_iterations
                                    master_time = stats.master_time if hasattr(stats, 'master_time') else 0
                                    subproblem_time = stats.subproblem_time if hasattr(stats, 'subproblem_time') else 0
                                    
                                    # Calculate MIP gap
                                    if objective > 0:
                                        mip_gap = (objective - lower_bound) / objective * 100  # As percentage
                                    else:
                                        mip_gap = 0.0 if abs(objective - lower_bound) < 1e-6 else float('inf')
                                    
                                    # Extract solution details
                                    solution = solver.best_solution
                                    
                                except TimeoutException:
                                    print(f"  Timeout occurred after {timeout} seconds.")
                                    
                                    # Create a result indicating timeout
                                    result = {
                                        **problem_info,
                                        'method': method,
                                        'status': 'timeout',
                                        'total_time': timeout,
                                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    results.append(result)
                                    continue
                                
                            elif method == "ml_benders":
                                # Use the real ML-enhanced Benders Decomposition
                                solver = MLBendersDecomposition(
                                    params, 
                                    use_ml=True,
                                    use_trust_region=False,
                                    use_laporte_cuts=False
                                )
                                
                                try:
                                    # Run with timeout
                                    lower_bound, objective, stats = run_with_timeout(
                                        solver.solve, 
                                        kwargs={'max_iterations': max_iterations},
                                        timeout_duration=timeout
                                    )
                                    
                                    subproblems = stats.num_iterations
                                    master_time = stats.master_time if hasattr(stats, 'master_time') else 0
                                    subproblem_time = stats.subproblem_time if hasattr(stats, 'subproblem_time') else 0
                                    ml_time = stats.ml_prediction_time if hasattr(stats, 'ml_prediction_time') else 0
                                    
                                    # Calculate MIP gap
                                    if objective > 0:
                                        mip_gap = (objective - lower_bound) / objective * 100  # As percentage
                                    else:
                                        mip_gap = 0.0 if abs(objective - lower_bound) < 1e-6 else float('inf')
                                    
                                    # Extract solution details
                                    solution = solver.best_solution
                                    
                                except TimeoutException:
                                    print(f"  Timeout occurred after {timeout} seconds.")
                                    
                                    # Create a result indicating timeout
                                    result = {
                                        **problem_info,
                                        'method': method,
                                        'status': 'timeout',
                                        'total_time': timeout,
                                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    results.append(result)
                                    continue
                            
                            total_time = time.time() - start_time
                            
                            # Store direct solve results for OPT gap calculation
                            if method == "direct":
                                # Create a key for this problem instance
                                instance_key = (periods, scenarios, c_value, f_value, seed)
                                direct_results[instance_key] = objective
                                # Direct method has no OPT gap (it is the reference)
                                opt_gap = 0.0
                            else:
                                # For non-direct methods, calculate OPT gap against direct solve if available
                                instance_key = (periods, scenarios, c_value, f_value, seed)
                                if instance_key in direct_results:
                                    direct_obj = direct_results[instance_key]
                                    # Calculate optimality gap as percentage difference from direct solution
                                    if direct_obj != 0:
                                        opt_gap = abs(objective - direct_obj) / abs(direct_obj) * 100
                                    else:
                                        opt_gap = 0.0 if abs(objective - direct_obj) < 1e-6 else float('inf')
                                else:
                                    # If direct solve result not available yet, set to NaN
                                    opt_gap = float('nan')
                            
                            # Extract statistics about solution
                            avg_setups = 0
                            avg_inventory = 0
                            
                            # Calculate average setups and inventory
                            for s in range(len(solution.setup)):
                                avg_setups += sum(1 for x in solution.setup[s] if x) / len(solution.setup[s])
                                avg_inventory += sum(solution.inventory[s]) / len(solution.inventory[s])
                            
                            avg_setups /= len(solution.setup)
                            avg_inventory /= len(solution.inventory)
                            
                            result = {
                                **problem_info,
                                'method': method,
                                'objective': objective,
                                'lower_bound': lower_bound,
                                'mip_gap': mip_gap,
                                'opt_gap': opt_gap,
                                'total_time': total_time,
                                'master_time': master_time,
                                'subproblem_time': subproblem_time,
                                'subproblems': subproblems,
                                'avg_setups': avg_setups,
                                'avg_inventory': avg_inventory,
                                'setup_percentage': avg_setups * 100,
                                'status': 'success',
                                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Add ML-specific statistics if available
                            if method == "ml_benders":
                                result['ml_time'] = ml_time
                            
                            # Print performance statistics
                            if verbose:
                                print(f"  Objective: {objective:.2f}")
                                print(f"  Lower Bound: {lower_bound:.2f}")
                                print(f"  MIP Gap: {mip_gap:.2f}%")
                                if method != "direct":  # Only show OPT gap for non-direct methods
                                    print(f"  OPT Gap: {opt_gap:.2f}%")
                                print(f"  Total time: {total_time:.2f}s")
                                print(f"  Subproblems/Iterations: {subproblems}")
                                print(f"  Avg. setups per period: {avg_setups:.2f} ({avg_setups*100:.1f}%)")
                                print(f"  Avg. inventory: {avg_inventory:.2f}")
                            
                        except Exception as e:
                            print(f"  Error: {str(e)}")
                            result = {
                                **problem_info,
                                'method': method,
                                'status': 'failed',
                                'error': str(e),
                                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        
                        results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_dir: str, report_name: str = None):
    """Save benchmark results to CSV and pickle files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        report_name: Custom name for the report files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate default report name if not provided
    if report_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"benchmark_{timestamp}"
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{report_name}.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Save as pickle
    pickle_path = os.path.join(output_dir, f"{report_name}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {pickle_path}")
    
    return results_df


def generate_summary_table(results_df: pd.DataFrame):
    """Generate and print a summary table of results.
    
    Args:
        results_df: DataFrame with benchmark results
        
    Returns:
        DataFrame with summary statistics
    """
    # Filter successful runs
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    # Add timeout information
    timeout_df = results_df[results_df['status'] == 'timeout'].copy()
    if len(timeout_df) > 0:
        print(f"\nNumber of timeouts: {len(timeout_df)}")
        timeout_summary = timeout_df.groupby(['periods', 'scenarios', 'c_value', 'f_value', 'method']).size().reset_index(name='count')
        print("\nTimeout Summary:")
        print("=" * 80)
        print(f"{'Periods':^8} | {'Scenarios':^9} | {'C':^3} | {'F':^7} | {'Method':^12} | {'Timeouts':^8}")
        print("-" * 80)
        
        for _, row in timeout_summary.iterrows():
            print(f"{row['periods']:^8} | {row['scenarios']:^9} | {row['c_value']:^3} | {row['f_value']:^7} | {row['method']:^12} | {row['count']:^8}")
    
    if len(success_df) == 0:
        print("No successful runs to analyze!")
        return None
    
    # Create base statistics DataFrame with manual calculation
    base_stats_dict = {}
    
    # Group by the keys we need
    grouped = success_df.groupby(['periods', 'scenarios', 'c_value', 'f_value', 'method'])
    
    # Manually calculate statistics to avoid MultiIndex problems
    base_stats_dict['total_time_mean'] = grouped['total_time'].mean()
    base_stats_dict['total_time_std'] = grouped['total_time'].std()
    base_stats_dict['total_time_min'] = grouped['total_time'].min()
    base_stats_dict['total_time_max'] = grouped['total_time'].max()
    
    base_stats_dict['subproblems_mean'] = grouped['subproblems'].mean()
    base_stats_dict['subproblems_min'] = grouped['subproblems'].min()
    base_stats_dict['subproblems_max'] = grouped['subproblems'].max()
    
    base_stats_dict['objective_mean'] = grouped['objective'].mean()
    base_stats_dict['objective_std'] = grouped['objective'].std()
    
    base_stats_dict['setup_percentage'] = grouped['setup_percentage'].mean()
    base_stats_dict['avg_inventory'] = grouped['avg_inventory'].mean()
    
    # Add MIP gap if it exists
    if 'mip_gap' in success_df.columns:
        base_stats_dict['mip_gap_mean'] = grouped['mip_gap'].mean()
        base_stats_dict['mip_gap_min'] = grouped['mip_gap'].min()
        base_stats_dict['mip_gap_max'] = grouped['mip_gap'].max()
    
    # Add OPT gap if it exists
    if 'opt_gap' in success_df.columns:
        base_stats_dict['opt_gap_mean'] = grouped['opt_gap'].mean()
        base_stats_dict['opt_gap_min'] = grouped['opt_gap'].min()
        base_stats_dict['opt_gap_max'] = grouped['opt_gap'].max()
    
    # Convert the dictionary to a DataFrame
    summary = pd.DataFrame(base_stats_dict)
    summary = summary.reset_index()
    
    # Print summary table
    print("\nPerformance Summary (Successful Runs Only):")
    print("=" * 170)
    print(f"{'Periods':^8} | {'Scenarios':^9} | {'C':^3} | {'F':^7} | {'Method':^12} | {'Avg Time (s)':^12} | {'Avg Subprobs':^12} | {'Obj Value':^12} | {'MIP Gap (%)':^15} | {'OPT Gap (%)':^15} | {'Setup %':^8} | {'Avg Inv':^8}")
    print("-" * 170)
    
    # Sort by problem size, C, F, and method
    summary = summary.sort_values(['periods', 'scenarios', 'c_value', 'f_value', 'method'])
    
    for _, row in summary.iterrows():
        periods = row['periods']
        scenarios = row['scenarios']
        c_value = row['c_value']
        f_value = row['f_value']
        method = row['method']
        
        avg_time = row['total_time_mean']
        avg_subprobs = row['subproblems_mean']
        avg_obj = row['objective_mean']
        
        # Get MIP gap and OPT gap with safe fallbacks
        try:
            avg_mip_gap = row['mip_gap_mean'] if 'mip_gap_mean' in row else float('nan')
            mip_gap_min = row['mip_gap_min'] if 'mip_gap_min' in row else float('nan')
            mip_gap_max = row['mip_gap_max'] if 'mip_gap_max' in row else float('nan')
        except (KeyError, TypeError):
            avg_mip_gap = float('nan')
            mip_gap_min = float('nan')
            mip_gap_max = float('nan')
            
        try:
            avg_opt_gap = row['opt_gap_mean'] if 'opt_gap_mean' in row else float('nan')
            opt_gap_min = row['opt_gap_min'] if 'opt_gap_min' in row else float('nan')
            opt_gap_max = row['opt_gap_max'] if 'opt_gap_max' in row else float('nan')
        except (KeyError, TypeError):
            avg_opt_gap = float('nan')
            opt_gap_min = float('nan')
            opt_gap_max = float('nan')
        
        setup_pct = row['setup_percentage']
        avg_inv = row['avg_inventory']
        
        # Format MIP gap display
        if pd.isna(avg_mip_gap):
            mip_gap_display = "N/A"
        else:
            mip_gap_display = f"{avg_mip_gap:.2f}%"
            if not pd.isna(mip_gap_min) and not pd.isna(mip_gap_max):
                mip_gap_display += f" ({mip_gap_min:.2f}%-{mip_gap_max:.2f}%)"
        
        # Format OPT gap display for direct method (where it's not applicable)
        if method == "direct" or pd.isna(avg_opt_gap):
            opt_gap_display = "N/A"
        else:
            opt_gap_display = f"{avg_opt_gap:.2f}%"
            if not pd.isna(opt_gap_min) and not pd.isna(opt_gap_max):
                opt_gap_display += f" ({opt_gap_min:.2f}%-{opt_gap_max:.2f}%)"
        
        print(f"{periods:^8} | {scenarios:^9} | {c_value:^3} | {f_value:^7} | {method:^12} | {avg_time:^12.2f} | {avg_subprobs:^12.1f} | {avg_obj:^12.2f} | {mip_gap_display:^15} | {opt_gap_display:^15} | {setup_pct:^8.1f} | {avg_inv:^8.1f}")
    
    return summary


def generate_method_comparison(results_df: pd.DataFrame, output_dir: str, report_name: str = None):
    """Generate method comparison charts and save them to files.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save results
        report_name: Custom name for the report files
        
    Returns:
        Directory path where plots were saved
    """
    # Filter successful runs
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("No successful runs to analyze!")
        return None
    
    methods = success_df['method'].unique()
    # Ensure we have a list
    if hasattr(methods, 'tolist'):
        methods = methods.tolist()
    
    if len(methods) <= 1:
        print("At least two successful methods needed for comparison!")
        return None
    
    # Create a combined problem size label
    success_df['size_label'] = success_df.apply(
        lambda row: f"({row['periods']}, {row['scenarios']})", axis=1)
    
    # Create a combined C,F parameter label
    success_df['param_label'] = success_df.apply(
        lambda row: f"C={row['c_value']}, F={row['f_value']}", axis=1)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Method colors and markers for consistent styling
    method_styles = {
        'direct': {'color': 'blue', 'marker': 'o', 'name': 'Direct Solver'},
        'benders': {'color': 'green', 'marker': 's', 'name': 'Benders Decomposition'},
        'ml_benders': {'color': 'red', 'marker': '^', 'name': 'ML-Enhanced Benders'}
    }
    
    # Group by parameters (C, F) to create separate comparisons
    for c_value in success_df['c_value'].unique():
        for f_value in success_df['f_value'].unique():
            param_df = success_df[(success_df['c_value'] == c_value) & (success_df['f_value'] == f_value)]
            
            if len(param_df) == 0:
                continue
                
            # Group by problem size and method, compute averages
            agg_funcs = {'total_time': 'mean', 'subproblems': 'mean', 'problem_size': 'first'}
            if 'mip_gap' in param_df.columns:
                agg_funcs['mip_gap'] = 'mean'
            if 'opt_gap' in param_df.columns:
                agg_funcs['opt_gap'] = 'mean'
                
            grouped = param_df.groupby(['periods', 'scenarios', 'method', 'size_label']).agg(agg_funcs).reset_index()
            
            # Sort by problem size
            grouped = grouped.sort_values('problem_size')
            
            # Get unique problem sizes in order
            size_labels = grouped['size_label'].unique()
            
            param_suffix = f"_C{c_value}_F{f_value}"
            
            # 1. Solution Time Comparison by Problem Size
            plt.figure(figsize=(12, 6))
            
            for method in methods:
                method_data = grouped[grouped['method'] == method]
                if len(method_data) > 0:
                    plt.plot(
                        method_data['size_label'], 
                        method_data['total_time'],
                        marker=method_styles[method]['marker'],
                        color=method_styles[method]['color'],
                        label=method_styles[method]['name']
                    )
            
            plt.title(f'Solution Time Comparison (C={c_value}, F={f_value})', fontsize=14)
            plt.xlabel('Problem Size (Periods, Scenarios)', fontsize=12)
            plt.ylabel('Average Solution Time (seconds)', fontsize=12)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            time_plot_path = os.path.join(plots_dir, f"{report_name}_time_comparison{param_suffix}.png")
            plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to avoid warning about too many open figures
            
            # 2. MIP Gap Comparison by Problem Size (if available)
            if 'mip_gap' in grouped.columns and not grouped['mip_gap'].isna().all():
                plt.figure(figsize=(12, 6))
                
                for method in methods:
                    method_data = grouped[grouped['method'] == method]
                    if len(method_data) > 0 and not method_data['mip_gap'].isna().all():
                        plt.plot(
                            method_data['size_label'], 
                            method_data['mip_gap'],
                            marker=method_styles[method]['marker'],
                            color=method_styles[method]['color'],
                            label=method_styles[method]['name']
                        )
                
                plt.title(f'MIP Gap Comparison (C={c_value}, F={f_value})', fontsize=14)
                plt.xlabel('Problem Size (Periods, Scenarios)', fontsize=12)
                plt.ylabel('Average MIP Gap (%)', fontsize=12)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                gap_plot_path = os.path.join(plots_dir, f"{report_name}_mip_gap_comparison{param_suffix}.png")
                plt.savefig(gap_plot_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to avoid warning about too many open figures
            
            # 3. OPT Gap Comparison for Benders methods (if available)
            if 'opt_gap' in grouped.columns and not grouped['opt_gap'].isna().all():
                benders_methods = [m for m in methods if 'benders' in m]
                
                if len(benders_methods) > 0:
                    plt.figure(figsize=(12, 6))
                    
                    for method in benders_methods:
                        method_data = grouped[grouped['method'] == method]
                        if len(method_data) > 0 and not method_data['opt_gap'].isna().all():
                            plt.plot(
                                method_data['size_label'], 
                                method_data['opt_gap'],
                                marker=method_styles[method]['marker'],
                                color=method_styles[method]['color'],
                                label=method_styles[method]['name']
                            )
                    
                    plt.title(f'OPT Gap Comparison - Benders Methods Only (C={c_value}, F={f_value})', fontsize=14)
                    plt.xlabel('Problem Size (Periods, Scenarios)', fontsize=12)
                    plt.ylabel('Average OPT Gap (%)', fontsize=12)
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save plot
                    opt_gap_plot_path = os.path.join(plots_dir, f"{report_name}_opt_gap_comparison{param_suffix}.png")
                    plt.savefig(opt_gap_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()  # Close the figure to avoid warning about too many open figures
            
            # 4. Subproblems Comparison (only for Benders methods)
            benders_methods = [m for m in methods if 'benders' in m]
            
            if len(benders_methods) > 0:
                plt.figure(figsize=(12, 6))
                
                for method in benders_methods:
                    method_data = grouped[grouped['method'] == method]
                    if len(method_data) > 0:
                        plt.plot(
                            method_data['size_label'], 
                            method_data['subproblems'],
                            marker=method_styles[method]['marker'],
                            color=method_styles[method]['color'],
                            label=method_styles[method]['name']
                        )
                
                plt.title(f'Number of Subproblems/Iterations Comparison (C={c_value}, F={f_value})', fontsize=14)
                plt.xlabel('Problem Size (Periods, Scenarios)', fontsize=12)
                plt.ylabel('Average Number of Subproblems/Iterations', fontsize=12)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                subprob_plot_path = os.path.join(plots_dir, f"{report_name}_subproblems_comparison{param_suffix}.png")
                plt.savefig(subprob_plot_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to avoid warning about too many open figures
    
    # Also create plots comparing different C, F values for each problem size and method
    # Group by problem size to create separate comparisons
    for periods in success_df['periods'].unique():
        for scenarios in success_df['scenarios'].unique():
            size_df = success_df[(success_df['periods'] == periods) & (success_df['scenarios'] == scenarios)]
            
            if len(size_df) == 0:
                continue
                
            for method in methods:
                method_df = size_df[size_df['method'] == method]
                
                if len(method_df) == 0:
                    continue
                    
                # Group by C and F parameters, compute averages
                grouped = method_df.groupby(['c_value', 'f_value', 'param_label']).agg({
                    'total_time': 'mean',
                    'subproblems': 'mean',
                }).reset_index()
                
                # Add MIP gap if it exists
                if 'mip_gap' in method_df.columns:
                    mip_gaps = method_df.groupby(['c_value', 'f_value', 'param_label'])['mip_gap'].mean().reset_index()
                    grouped = pd.merge(grouped, mip_gaps, on=['c_value', 'f_value', 'param_label'], how='left')
                
                # Add OPT gap if it exists (for Benders methods)
                if 'opt_gap' in method_df.columns and 'benders' in method:
                    opt_gaps = method_df.groupby(['c_value', 'f_value', 'param_label'])['opt_gap'].mean().reset_index()
                    grouped = pd.merge(grouped, opt_gaps, on=['c_value', 'f_value', 'param_label'], how='left')
                
                # Sort by C and F values
                grouped = grouped.sort_values(['c_value', 'f_value'])
                
                size_suffix = f"_P{periods}_S{scenarios}_{method}"
                
                # 1. Solution Time Comparison by C,F Parameters
                plt.figure(figsize=(12, 6))
                
                # Grouped bar chart for C,F parameters
                bar_width = 0.25
                num_c_values = len(grouped['c_value'].unique())
                x = np.arange(len(grouped['f_value'].unique()))
                
                for i, c_value in enumerate(sorted(grouped['c_value'].unique())):
                    c_data = grouped[grouped['c_value'] == c_value]
                    plt.bar(
                        x + i*bar_width, 
                        c_data['total_time'],
                        width=bar_width,
                        label=f'C={c_value}'
                    )
                
                plt.title(f'Solution Time by C,F Parameters ({periods} periods, {scenarios} scenarios, {method})', fontsize=14)
                plt.xlabel('F Value', fontsize=12)
                plt.ylabel('Average Solution Time (seconds)', fontsize=12)
                plt.xticks(x + bar_width*(num_c_values-1)/2, sorted(grouped['f_value'].unique()))
                plt.legend()
                plt.tight_layout()
                
                # Save plot
                param_plot_path = os.path.join(plots_dir, f"{report_name}_param_comparison{size_suffix}.png")
                plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to avoid warning about too many open figures
    
    print(f"Comparison plots saved to {plots_dir}")
    
    return plots_dir


def generate_html_report(
    results_df: pd.DataFrame, 
    output_dir: str, 
    report_name: str = None,
    plots_dir: str = None
):
    """Generate a comprehensive HTML report.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save results
        report_name: Custom name for the report
        plots_dir: Directory with generated plots
        
    Returns:
        Path to the generated HTML report
    """
    # Generate default report name if not provided
    if report_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"benchmark_{timestamp}"
    
    # Filter successful runs
    success_df = results_df[results_df['status'] == 'success'].copy()
    timeout_df = results_df[results_df['status'] == 'timeout'].copy()
    
    if len(success_df) == 0 and len(timeout_df) == 0:
        print("No runs to analyze!")
        return None
    
    # Create base statistics DataFrame with manual calculation
    base_stats_dict = {}
    
    # Group by the keys we need
    if len(success_df) > 0:
        grouped = success_df.groupby(['periods', 'scenarios', 'c_value', 'f_value', 'method'])
        
        # Manually calculate statistics to avoid MultiIndex problems
        base_stats_dict['total_time_mean'] = grouped['total_time'].mean()
        base_stats_dict['total_time_std'] = grouped['total_time'].std()
        base_stats_dict['total_time_min'] = grouped['total_time'].min()
        base_stats_dict['total_time_max'] = grouped['total_time'].max()
        
        base_stats_dict['subproblems_mean'] = grouped['subproblems'].mean()
        base_stats_dict['subproblems_min'] = grouped['subproblems'].min()
        base_stats_dict['subproblems_max'] = grouped['subproblems'].max()
        
        base_stats_dict['objective_mean'] = grouped['objective'].mean()
        base_stats_dict['objective_std'] = grouped['objective'].std()
        
        base_stats_dict['setup_percentage'] = grouped['setup_percentage'].mean()
        base_stats_dict['avg_inventory'] = grouped['avg_inventory'].mean()
        
        # Add MIP gap if it exists
        if 'mip_gap' in success_df.columns:
            base_stats_dict['mip_gap_mean'] = grouped['mip_gap'].mean()
            base_stats_dict['mip_gap_min'] = grouped['mip_gap'].min()
            base_stats_dict['mip_gap_max'] = grouped['mip_gap'].max()
        
        # Add OPT gap if it exists
        if 'opt_gap' in success_df.columns:
            base_stats_dict['opt_gap_mean'] = grouped['opt_gap'].mean()
            base_stats_dict['opt_gap_min'] = grouped['opt_gap'].min()
            base_stats_dict['opt_gap_max'] = grouped['opt_gap'].max()
        
        # Convert the dictionary to a DataFrame
        summary = pd.DataFrame(base_stats_dict)
        summary = summary.reset_index()
        
        # Sort by problem size and method
        summary = summary.sort_values(['periods', 'scenarios', 'c_value', 'f_value', 'method'])
    else:
        summary = pd.DataFrame()
    
    # Count timeouts by problem configuration
    if len(timeout_df) > 0:
        timeout_counts = timeout_df.groupby(['periods', 'scenarios', 'c_value', 'f_value', 'method']).size().reset_index(name='timeouts')
    else:
        timeout_counts = pd.DataFrame()
    
    # Get overall statistics
    total_problems = len(results_df)
    solved_problems = len(success_df)
    timeout_problems = len(timeout_df)
    
    if len(success_df) > 0:
        avg_time = success_df['total_time'].mean()
    else:
        avg_time = float('nan')
    
    # Methods used
    methods = results_df['method'].unique()
    # Ensure we have a list
    if hasattr(methods, 'tolist'):
        methods = methods.tolist()
    methods_str = ", ".join([m.replace('_', '-').title() for m in methods])
    
    # Problem sizes
    size_data = results_df[['periods', 'scenarios']].drop_duplicates()
    problem_sizes = []
    for _, row in size_data.iterrows():
        period_val = row['periods']
        scenario_val = row['scenarios']
        problem_sizes.append(f"({period_val}, {scenario_val})")
        
    sizes_str = ", ".join(problem_sizes)
    
    # C, F parameters
    c_values = sorted(results_df['c_value'].unique())
    f_values = sorted(results_df['f_value'].unique())
    
    # Format as lists
    c_str = ", ".join([str(c) for c in c_values])
    f_str = ", ".join([str(f) for f in f_values])
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stochastic Production Planning Benchmark Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
                margin-top: 30px;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .summary-box {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                text-align: right;
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #eaf2f8;
            }}
            .method-comparison {{
                margin-top: 40px;
            }}
            .chart-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 0.9em;
                text-align: center;
                color: #7f8c8d;
            }}
            .timeout {{
                color: #e74c3c;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>Stochastic Production Planning Benchmark Report</h1>
        
        <div class="summary-box">
            <h2>Benchmark Summary</h2>
            <p><strong>Date:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            <p><strong>Total problems attempted:</strong> {total_problems}</p>
            <p><strong>Successfully solved:</strong> {solved_problems}</p>
            <p><strong>Timeouts:</strong> <span class="timeout">{timeout_problems}</span></p>
            <p><strong>Methods compared:</strong> {methods_str}</p>
            <p><strong>Problem sizes:</strong> {sizes_str}</p>
            <p><strong>C values (capacity-to-demand ratios):</strong> {c_str}</p>
            <p><strong>F values (setup-to-holding cost ratios):</strong> {f_str}</p>
    """
    
    if len(success_df) > 0:
        html_content += f"""
            <p><strong>Average solution time (successful runs):</strong> {avg_time:.2f} seconds</p>
        """
    
    html_content += """
        </div>
    """
    
    # Add timeout summary if there were timeouts
    if len(timeout_df) > 0:
        html_content += """
        <h2>Timeout Summary</h2>
        <table>
            <tr>
                <th>Periods</th>
                <th>Scenarios</th>
                <th>C</th>
                <th>F</th>
                <th>Method</th>
                <th>Number of Timeouts</th>
            </tr>
        """
        
        for _, row in timeout_counts.iterrows():
            html_content += f"""
            <tr>
                <td>{row['periods']}</td>
                <td>{row['scenarios']}</td>
                <td>{row['c_value']}</td>
                <td>{row['f_value']}</td>
                <td>{row['method'].replace('_', '-').title()}</td>
                <td class="timeout">{row['timeouts']}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    if len(success_df) > 0:
        html_content += """
        <h2>Performance by Problem Size, Parameters, and Method (Successful Runs Only)</h2>
        <table>
            <tr>
                <th>Periods</th>
                <th>Scenarios</th>
                <th>C</th>
                <th>F</th>
                <th>Method</th>
                <th>Avg Time (s)</th>
                <th>Min Time</th>
                <th>Max Time</th>
                <th>Avg Subproblems</th>
                <th>Avg Objective</th>
                <th>MIP Gap (%)</th>
                <th>OPT Gap (%)</th>
                <th>Setup %</th>
                <th>Avg Inventory</th>
            </tr>
        """
        
        # Add rows for each problem size and method
        for _, row in summary.iterrows():
            periods = row['periods']
            scenarios = row['scenarios']
            c_value = row['c_value']
            f_value = row['f_value']
            method = row['method']
                
            # Format method name for display
            method_display = method.replace('_', '-').title()
            
            time_mean = row['total_time_mean']
            time_min = row['total_time_min']
            time_max = row['total_time_max']
            
            subprob_mean = row['subproblems_mean']
            obj_mean = row['objective_mean']
            
            # Get MIP gap with safe fallbacks
            try:
                mip_gap_mean = row['mip_gap_mean'] if 'mip_gap_mean' in row else float('nan')
                mip_gap_min = row['mip_gap_min'] if 'mip_gap_min' in row else float('nan')
                mip_gap_max = row['mip_gap_max'] if 'mip_gap_max' in row else float('nan')
            except (KeyError, TypeError):
                mip_gap_mean = float('nan')
                mip_gap_min = float('nan')
                mip_gap_max = float('nan')
            
            # Get OPT gap with safe fallbacks
            try:
                opt_gap_mean = row['opt_gap_mean'] if 'opt_gap_mean' in row else float('nan')
                opt_gap_min = row['opt_gap_min'] if 'opt_gap_min' in row else float('nan')
                opt_gap_max = row['opt_gap_max'] if 'opt_gap_max' in row else float('nan')
            except (KeyError, TypeError):
                opt_gap_mean = float('nan')
                opt_gap_min = float('nan')
                opt_gap_max = float('nan')
            
            setup_mean = row['setup_percentage']
            inv_mean = row['avg_inventory']
            
            # Format MIP gap for display
            if pd.isna(mip_gap_mean):
                mip_gap_display = "N/A"
            else:
                mip_gap_display = f"{mip_gap_mean:.2f}%"
                if not pd.isna(mip_gap_min) and not pd.isna(mip_gap_max):
                    mip_gap_display += f" ({mip_gap_min:.2f}%-{mip_gap_max:.2f}%)"
            
            # Format OPT gap for display (N/A for direct method)
            if method == "direct" or pd.isna(opt_gap_mean):
                opt_gap_display = "N/A"
            else:
                opt_gap_display = f"{opt_gap_mean:.2f}%"
                if not pd.isna(opt_gap_min) and not pd.isna(opt_gap_max):
                    opt_gap_display += f" ({opt_gap_min:.2f}%-{opt_gap_max:.2f}%)"
            
            html_content += f"""
                <tr>
                    <td>{periods}</td>
                    <td>{scenarios}</td>
                    <td>{c_value}</td>
                    <td>{f_value}</td>
                    <td>{method_display}</td>
                    <td>{time_mean:.2f}</td>
                    <td>{time_min:.2f}</td>
                    <td>{time_max:.2f}</td>
                    <td>{subprob_mean:.1f}</td>
                    <td>{obj_mean:.2f}</td>
                    <td>{mip_gap_display}</td>
                    <td>{opt_gap_display}</td>
                    <td>{setup_mean:.1f}</td>
                    <td>{inv_mean:.1f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2 class="method-comparison">Method Comparison</h2>
        """
    
    # Add visualizations if available
    if plots_dir and len(success_df) > 0:
        # Get relative paths to plots
        plots_rel_path = os.path.relpath(plots_dir, output_dir)
        
        # Find all the C and F value combinations in the plots directory
        c_f_patterns = []
        for c_value in c_values:
            for f_value in f_values:
                c_f_patterns.append(f"_C{c_value}_F{f_value}")
        
        # For each C and F combination, add the relevant plots
        for param_suffix in c_f_patterns:
            # Extract C and F values from the pattern
            c_value = param_suffix.split('_C')[1].split('_F')[0]
            f_value = param_suffix.split('_F')[1]
            
            # Check if time comparison exists
            time_plot = f"{plots_rel_path}/{report_name}_time_comparison{param_suffix}.png"
            if os.path.exists(os.path.join(output_dir, time_plot)):
                html_content += f"""
                <h3>Results for C={c_value}, F={f_value}</h3>
                
                <div class="chart-container">
                    <h4>Solution Time Comparison</h4>
                    <img src="{time_plot}" alt="Solution Time Comparison" style="max-width: 100%; height: auto;">
                </div>
                """
            else:
                continue  # Skip this C,F combination if no plots exist
            
            # Check if MIP gap comparison exists
            mip_gap_plot = f"{plots_rel_path}/{report_name}_mip_gap_comparison{param_suffix}.png"
            if os.path.exists(os.path.join(output_dir, mip_gap_plot)):
                html_content += f"""
                <div class="chart-container">
                    <h4>MIP Gap Comparison</h4>
                    <img src="{mip_gap_plot}" alt="MIP Gap Comparison" style="max-width: 100%; height: auto;">
                </div>
                """
            
            # Check if OPT gap comparison exists
            opt_gap_plot = f"{plots_rel_path}/{report_name}_opt_gap_comparison{param_suffix}.png"
            if os.path.exists(os.path.join(output_dir, opt_gap_plot)):
                html_content += f"""
                <div class="chart-container">
                    <h4>OPT Gap Comparison (Benders Methods Only)</h4>
                    <img src="{opt_gap_plot}" alt="OPT Gap Comparison" style="max-width: 100%; height: auto;">
                </div>
                """
            
            # Check if subproblems comparison exists
            subprob_plot = f"{plots_rel_path}/{report_name}_subproblems_comparison{param_suffix}.png"
            if os.path.exists(os.path.join(output_dir, subprob_plot)):
                html_content += f"""
                <div class="chart-container">
                    <h4>Subproblems/Iterations Comparison (Benders Methods)</h4>
                    <img src="{subprob_plot}" alt="Subproblems Comparison" style="max-width: 100%; height: auto;">
                </div>
                """
        
        # Add comparison of C,F parameters for each problem size and method
        html_content += """
        <h3>Parameter Impact Analysis</h3>
        """
        
        # Find parameter comparison plots
        for periods in success_df['periods'].unique():
            for scenarios in success_df['scenarios'].unique():
                for method in methods:
                    param_plot = f"{plots_rel_path}/{report_name}_param_comparison_P{periods}_S{scenarios}_{method}.png"
                    if os.path.exists(os.path.join(output_dir, param_plot)):
                        html_content += f"""
                        <div class="chart-container">
                            <h4>C,F Parameter Impact for {periods} periods, {scenarios} scenarios, {method.replace('_', '-').title()}</h4>
                            <img src="{param_plot}" alt="Parameter Comparison" style="max-width: 100%; height: auto;">
                        </div>
                        """
    
    # Add raw data table for all runs
    html_content += """
        <h2>Raw Benchmark Data</h2>
        <table>
            <tr>
                <th>Periods</th>
                <th>Scenarios</th>
                <th>C</th>
                <th>F</th>
                <th>Method</th>
                <th>Seed</th>
                <th>Status</th>
                <th>Total Time (s)</th>
                <th>Subproblems</th>
                <th>Objective Value</th>
                <th>MIP Gap (%)</th>
                <th>OPT Gap (%)</th>
            </tr>
    """
    
    # Sort by problem size and method
    sorted_df = results_df.sort_values(['periods', 'scenarios', 'c_value', 'f_value', 'method', 'seed'])
    
    # Add rows for each individual run
    for _, row in sorted_df.iterrows():
        # Extract values directly
        periods = row['periods']
        scenarios = row['scenarios']
        c_value = row['c_value']
        f_value = row['f_value']
        method = row['method']
        seed = row['seed']
        status = row['status']
        
        # Format method name for display
        method_display = method.replace('_', '-').title()
        
        # Format status with color for timeouts
        status_class = ' class="timeout"' if status == 'timeout' else ''
        
        # Get values based on status
        if status == 'success':
            total_time = row['total_time']
            subproblems = row['subproblems']
            objective = row['objective']
            
            # Safely get MIP gap and OPT gap
            mip_gap = row['mip_gap'] if 'mip_gap' in row else float('nan')
            opt_gap = row['opt_gap'] if 'opt_gap' in row else float('nan')
            
            # Format MIP gap for display
            if pd.isna(mip_gap):
                mip_gap_display = "N/A"
            else:
                mip_gap_display = f"{mip_gap:.2f}"
            
            # Format OPT gap for display (N/A for direct method)
            if method == "direct" or pd.isna(opt_gap):
                opt_gap_display = "N/A"
            else:
                opt_gap_display = f"{opt_gap:.2f}"
                
            html_content += f"""
                <tr>
                    <td>{periods}</td>
                    <td>{scenarios}</td>
                    <td>{c_value}</td>
                    <td>{f_value}</td>
                    <td>{method_display}</td>
                    <td>{seed}</td>
                    <td>{status}</td>
                    <td>{total_time:.2f}</td>
                    <td>{subproblems}</td>
                    <td>{objective:.2f}</td>
                    <td>{mip_gap_display}</td>
                    <td>{opt_gap_display}</td>
                </tr>
            """
        elif status == 'timeout':
            html_content += f"""
                <tr>
                    <td>{periods}</td>
                    <td>{scenarios}</td>
                    <td>{c_value}</td>
                    <td>{f_value}</td>
                    <td>{method_display}</td>
                    <td>{seed}</td>
                    <td{status_class}>{status.upper()}</td>
                    <td{status_class}>{'timeout'}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            """
        else:  # Failed or other status
            error_msg = row['error'] if 'error' in row else "Unknown error"
            html_content += f"""
                <tr>
                    <td>{periods}</td>
                    <td>{scenarios}</td>
                    <td>{c_value}</td>
                    <td>{f_value}</td>
                    <td>{method_display}</td>
                    <td>{seed}</td>
                    <td>FAILED</td>
                    <td colspan="5">{error_msg}</td>
                </tr>
            """
    
    # Close the table and add footer
    html_content += f"""
        </table>
        
        <div class="footer">
            <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report to file
    report_path = os.path.join(output_dir, f"{report_name}.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML report saved to {report_path}")
    
    return report_path


def main():
    """Main function to run benchmarks and generate reports."""
    # Parse command line arguments
    args = parse_args()
    
    # Parse problem sizes
    problem_sizes = parse_problem_sizes(args.problem_sizes)
    
    # Parse seeds
    seeds = list(map(int, args.seeds.split(",")))
    
    # Parse methods
    methods = args.methods.split(",")
    valid_methods = ['direct', 'benders', 'ml_benders']
    
    # Parse C and F values
    c_values = list(map(int, args.c_values.split(",")))
    f_values = list(map(int, args.f_values.split(",")))
    
    # Validate methods
    for method in methods:
        if method not in valid_methods:
            print(f"Error: Invalid method '{method}'. Valid options are: {', '.join(valid_methods)}")
            sys.exit(1)
    
    # Generate a report name if not provided
    if args.report_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"benchmark_{timestamp}"
    else:
        report_name = args.report_name
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print benchmark configuration
    print("=" * 80)
    print("Benchmark Configuration:")
    print("=" * 80)
    print(f"Problem sizes: {problem_sizes}")
    print(f"C values (capacity-to-demand ratios): {c_values}")
    print(f"F values (setup-to-holding cost ratios): {f_values}")
    print(f"Random seeds: {seeds}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Methods: {methods}")
    print(f"Timeout per solver: {args.timeout} seconds")
    print(f"Maximum iterations/subproblems: {args.max_iterations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Report name: {report_name}")
    print("=" * 80)
    print()
    
    # Run benchmarks
    start_time = time.time()
    
    results = run_benchmark(
        problem_sizes=problem_sizes,
        seeds=seeds,
        c_values=c_values,
        f_values=f_values,
        num_trials=args.num_trials,
        methods=methods,
        timeout=args.timeout,
        max_iterations=args.max_iterations,
        verbose=args.verbose
    )
    
    total_time = time.time() - start_time
    
    # Save results to CSV and pickle
    results_df = save_results(results, args.output_dir, report_name)
    
    # Generate summary table
    summary = generate_summary_table(results_df)
    
    # Generate method comparison plots if there are successful runs
    plots_dir = None
    if 'status' in results_df and (results_df['status'] == 'success').any():
        plots_dir = generate_method_comparison(results_df, args.output_dir, report_name)
    
    # Generate HTML report
    generate_html_report(results_df, args.output_dir, report_name, plots_dir)
    
    # Print summary
    print("\nBenchmark completed!")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()