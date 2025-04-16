#!/usr/bin/env python
"""
Performance evaluation for production planning Benders decomposition.

This script:
1. Loads generated data or creates test problems
2. Runs the Benders decomposition without ML on various problem sizes
3. Collects and reports performance metrics
4. Generates visualizations of the results
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm
import sys

# Add current directory to path to help with imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import needed modules using absolute imports
from data.data_structures import ScenarioData, ProblemParameters, TimingStats, Solution
from data.problem_generator import create_test_problem
from models.benders import MLBendersDecomposition

def run_benchmark(problem_sizes=None, seeds=None, num_trials=3, use_ml=False, output_dir=None):
    """Run benchmarks comparing ML-based approach with exact solution.
    
    Args:
        problem_sizes: List of tuples (periods, scenarios)
        seeds: List of random seeds
        num_trials: Number of trials to run
        use_ml: Whether to use ML enhancement
        output_dir: Directory to save results
        
    Returns:
        DataFrame with benchmark results
    """
    if problem_sizes is None:
        problem_sizes = [(12, 2), (24, 3), (36, 4)]
    
    if seeds is None:
        seeds = [42, 43, 44]
    
    results = []
    
    for periods, scenarios in problem_sizes:
        for seed in seeds[:num_trials]:
            print(f"\nBenchmarking problem with {periods} periods, {scenarios} scenarios (seed {seed})...")
            
            # Create problem
            params = create_test_problem(periods, scenarios, seed)
            
            # Record problem characteristics
            problem_info = {
                'periods': periods,
                'scenarios': scenarios,
                'seed': seed,
                'total_demand': sum(sum(s.demands) for s in params.scenarios) / len(params.scenarios),
                'demand_variability': np.std([item for s in params.scenarios for item in s.demands]),
                'use_ml': use_ml
            }
            
            # Solve with exact approach
            print("Solving with exact approach...")
            exact_decomp = MLBendersDecomposition(params, use_ml=use_ml)
            
            try:
                start_time = time.time()
                exact_lb, exact_ub, exact_stats = exact_decomp.solve(max_iterations=50)
                total_time = time.time() - start_time
                
                # Calculate MIP gap - relative difference between upper and lower bounds
                if exact_ub > 0:
                    mip_gap = (exact_ub - exact_lb) / exact_ub * 100  # As percentage
                else:
                    mip_gap = 0.0 if abs(exact_ub - exact_lb) < 1e-6 else float('inf')
                
                result = {
                    **problem_info,
                    'objective': exact_ub,
                    'lower_bound': exact_lb,
                    'mip_gap': mip_gap,
                    'total_time': total_time,
                    'subproblem_time': exact_stats.subproblem_time,
                    'master_time': exact_stats.master_time,
                    'iterations': exact_stats.num_iterations,
                    'status': 'success'
                }
                
                # Collect detailed solution statistics
                avg_setups = 0
                avg_inventory = 0
                for s in range(len(exact_decomp.best_solution.setup)):
                    avg_setups += sum(exact_decomp.best_solution.setup[s]) / len(exact_decomp.best_solution.setup[s])
                    avg_inventory += sum(exact_decomp.best_solution.inventory[s]) / len(exact_decomp.best_solution.inventory[s])
                
                avg_setups /= len(exact_decomp.best_solution.setup)
                avg_inventory /= len(exact_decomp.best_solution.inventory)
                
                result['avg_setups'] = avg_setups
                result['avg_inventory'] = avg_inventory
                result['setup_percentage'] = avg_setups * 100
                
                # Print performance statistics
                print(f"Objective: {exact_ub:.2f}")
                print(f"Lower Bound: {exact_lb:.2f}")
                print(f"MIP Gap: {mip_gap:.2f}%")
                print(f"Total time: {total_time:.2f}s")
                print(f"Iterations: {exact_stats.num_iterations}")
                print(f"Avg. setups per period: {avg_setups:.2f} ({avg_setups*100:.1f}%)")
                print(f"Avg. inventory: {avg_inventory:.2f}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                result = {
                    **problem_info,
                    'status': 'failed',
                    'error': str(e)
                }
            
            results.append(result)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ml_suffix = "_ml" if use_ml else "_exact"
        results_df.to_csv(os.path.join(output_dir, f"benchmark_results{ml_suffix}.csv"), index=False)
        
        # Also save as pickle for easier loading
        with open(os.path.join(output_dir, f"benchmark_results{ml_suffix}.pkl"), "wb") as f:
            pickle.dump(results, f)
    
    return results_df

def generate_performance_report(results_df, output_dir=None):
    """Generate a comprehensive performance report.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save report
    """
    # Filter successful runs
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("No successful runs to analyze!")
        return
    
    # Create summary statistics by problem size
    summary = success_df.groupby(['periods', 'scenarios']).agg({
        'total_time': ['mean', 'std', 'min', 'max'],
        'iterations': ['mean', 'std', 'min', 'max'],
        'objective': ['mean', 'std'],
        'mip_gap': ['mean', 'min', 'max'],
        'setup_percentage': ['mean'],
        'avg_inventory': ['mean']
    }).reset_index()
    
    # Print summary table
    print("\nPerformance Summary:")
    print("=" * 100)
    print(f"{'Periods':^8} | {'Scenarios':^9} | {'Avg Time (s)':^12} | {'Avg Iter':^8} | {'Obj Value':^12} | {'MIP Gap (%)':^10} | {'Setup %':^8} | {'Avg Inv':^8}")
    print("-" * 100)
    
    for i, row in summary.iterrows():
        periods = row['periods']
        scenarios = row['scenarios']
        
        # Access scalar values properly - important fix!
        avg_time = row[('total_time', 'mean')]
        avg_iter = row[('iterations', 'mean')]
        avg_obj = row[('objective', 'mean')]
        avg_gap = row[('mip_gap', 'mean')]
        setup_pct = row[('setup_percentage', 'mean')]
        avg_inv = row[('avg_inventory', 'mean')]
        
        # Handle scalar vs Series type to avoid formatting errors
        if hasattr(periods, 'iloc'):
            periods = periods.iloc[0]
        if hasattr(scenarios, 'iloc'):
            scenarios = scenarios.iloc[0]
        if hasattr(avg_time, 'iloc'):
            avg_time = avg_time.iloc[0]
        if hasattr(avg_iter, 'iloc'):
            avg_iter = avg_iter.iloc[0]
        if hasattr(avg_obj, 'iloc'):
            avg_obj = avg_obj.iloc[0]
        if hasattr(avg_gap, 'iloc'):
            avg_gap = avg_gap.iloc[0]
        if hasattr(setup_pct, 'iloc'):
            setup_pct = setup_pct.iloc[0]
        if hasattr(avg_inv, 'iloc'):
            avg_inv = avg_inv.iloc[0]
        
        print(f"{periods:^8} | {scenarios:^9} | {avg_time:^12.2f} | {avg_iter:^8.1f} | {avg_obj:^12.2f} | {avg_gap:^10.2f} | {setup_pct:^8.1f} | {avg_inv:^8.1f}")
    
    # If there aren't enough data points, skip visualizations
    if len(success_df) < 2:
        print("\nNot enough successful runs to create visualizations.")
        return
    
    try:
        # Create a combined size metric (periods * scenarios)
        success_df['problem_size'] = success_df['periods'] * success_df['scenarios']
        
        # Plot 1: Computation time vs problem size
        plt.figure(figsize=(10, 6))
        
        # Group by problem size and compute average
        size_time_df = success_df.groupby('problem_size').agg({
            'total_time': 'mean',
            'periods': 'first',
            'scenarios': 'first'
        }).reset_index()
        
        plt.scatter(size_time_df['problem_size'], size_time_df['total_time'], s=100, alpha=0.7)
        
        # Add labels with safe extraction for Series objects
        for _, row in size_time_df.iterrows():
            period_val = float(row['periods']) if hasattr(row['periods'], 'iloc') else row['periods']
            scenario_val = float(row['scenarios']) if hasattr(row['scenarios'], 'iloc') else row['scenarios']
            
            plt.annotate(f"({period_val:.0f}, {scenario_val:.0f})",
                        (row['problem_size'], row['total_time']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.xlabel('Problem Size (Periods × Scenarios)')
        plt.ylabel('Average Computation Time (s)')
        plt.title('Computation Time vs Problem Size')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'time_vs_size.png'), dpi=300, bbox_inches='tight')
        
        # Plot 2: Iterations vs problem size
        plt.figure(figsize=(10, 6))
        
        # Group by problem size and compute average iterations
        size_iter_df = success_df.groupby('problem_size').agg({
            'iterations': 'mean',
            'periods': 'first',
            'scenarios': 'first'
        }).reset_index()
        
        plt.scatter(size_iter_df['problem_size'], size_iter_df['iterations'], s=100, alpha=0.7)
        
        # Add labels with safe extraction for Series objects
        for _, row in size_iter_df.iterrows():
            period_val = float(row['periods']) if hasattr(row['periods'], 'iloc') else row['periods']
            scenario_val = float(row['scenarios']) if hasattr(row['scenarios'], 'iloc') else row['scenarios']
            
            plt.annotate(f"({period_val:.0f}, {scenario_val:.0f})",
                        (row['problem_size'], row['iterations']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.xlabel('Problem Size (Periods × Scenarios)')
        plt.ylabel('Average Number of Iterations')
        plt.title('Iterations vs Problem Size')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'iterations_vs_size.png'), dpi=300, bbox_inches='tight')
        
        # Plot 3: Time breakdown
        if 'subproblem_time' in success_df.columns and 'master_time' in success_df.columns:
            plt.figure(figsize=(12, 6))
            
            problem_sizes = sorted(success_df['problem_size'].unique())
            
            # Compute average times for each problem size
            subprob_times = []
            master_times = []
            labels = []
            
            for size in problem_sizes:
                size_df = success_df[success_df['problem_size'] == size]
                subprob_times.append(size_df['subproblem_time'].mean())
                master_times.append(size_df['master_time'].mean())
                
                # Get a representative row for this size
                rep_row = size_df.iloc[0]
                
                # Safe extraction for potentially Series objects
                period_val = float(rep_row['periods']) if hasattr(rep_row['periods'], 'iloc') else rep_row['periods']
                scenario_val = float(rep_row['scenarios']) if hasattr(rep_row['scenarios'], 'iloc') else rep_row['scenarios']
                
                labels.append(f"({period_val:.0f}, {scenario_val:.0f})")
            
            # Create stacked bar chart
            width = 0.6
            plt.bar(labels, subprob_times, width, label='Subproblem Time')
            plt.bar(labels, master_times, width, bottom=subprob_times, label='Master Problem Time')
            
            plt.xlabel('Problem Size (Periods, Scenarios)')
            plt.ylabel('Time (s)')
            plt.title('Time Breakdown by Problem Component')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'time_breakdown.png'), dpi=300, bbox_inches='tight')
        
        # Plot 4: MIP gap vs problem size
        if 'mip_gap' in success_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Group by problem size and compute average gap
            size_gap_df = success_df.groupby('problem_size').agg({
                'mip_gap': ['mean', 'min', 'max'],
                'periods': 'first',
                'scenarios': 'first'
            }).reset_index()
            
            # Extract mean, min, max gap values safely
            gap_means = []
            gap_mins = []
            gap_maxs = []
            
            for _, row in size_gap_df.iterrows():
                mean_gap = float(row[('mip_gap', 'mean')]) if hasattr(row[('mip_gap', 'mean')], 'iloc') else row[('mip_gap', 'mean')]
                min_gap = float(row[('mip_gap', 'min')]) if hasattr(row[('mip_gap', 'min')], 'iloc') else row[('mip_gap', 'min')]
                max_gap = float(row[('mip_gap', 'max')]) if hasattr(row[('mip_gap', 'max')], 'iloc') else row[('mip_gap', 'max')]
                
                gap_means.append(mean_gap)
                gap_mins.append(min_gap)
                gap_maxs.append(max_gap)
            
            # Plot mean gaps
            plt.scatter(size_gap_df['problem_size'], gap_means, s=100, alpha=0.7)
            
            # Add error bars for min/max
            plt.errorbar(
                size_gap_df['problem_size'], 
                gap_means,
                yerr=[
                    np.array(gap_means) - np.array(gap_mins),
                    np.array(gap_maxs) - np.array(gap_means)
                ],
                fmt='none', ecolor='gray', capsize=5
            )
            
            # Add labels with safe extraction
            for i, row in size_gap_df.iterrows():
                period_val = float(row['periods']) if hasattr(row['periods'], 'iloc') else row['periods']
                scenario_val = float(row['scenarios']) if hasattr(row['scenarios'], 'iloc') else row['scenarios']
                
                plt.annotate(f"({period_val:.0f}, {scenario_val:.0f})",
                           (row['problem_size'], gap_means[i]),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
            
            plt.xlabel('Problem Size (Periods × Scenarios)')
            plt.ylabel('MIP Gap (%)')
            plt.title('MIP Gap vs Problem Size')
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'gap_vs_size.png'), dpi=300, bbox_inches='tight')
    
    except Exception as e:
        print(f"\nError creating visualizations: {e}")
    
    # If we have output directory, save a full HTML report
    if output_dir:
        try:
            # Get mean values for summary
            avg_time = float(success_df['total_time'].mean())
            avg_iter = float(success_df['iterations'].mean())
            
            # Create HTML report with f-string to avoid format issues
            html_report = f"""
            <html>
            <head>
                <title>Benders Decomposition Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .chart-container {{ margin: 30px 0; text-align: center; }}
                    .summary {{ background-color: #eef8ff; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Benders Decomposition Performance Report</h1>
                
                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <p>Total problems solved: {len(success_df)}</p>
                    <p>Average solution time: {avg_time:.2f} seconds</p>
                    <p>Average iterations: {avg_iter:.1f}</p>
                </div>
                
                <h2>Performance by Problem Size</h2>
                <table>
                    <tr>
                        <th>Periods</th>
                        <th>Scenarios</th>
                        <th>Avg Time (s)</th>
                        <th>Std Time</th>
                        <th>Min Time</th>
                        <th>Max Time</th>
                        <th>Avg Iterations</th>
                        <th>Avg Objective</th>
                        <th>Avg MIP Gap</th>
                        <th>Setup %</th>
                        <th>Avg Inventory</th>
                    </tr>
            """
            
            # Add rows for each problem size
            for _, row in summary.iterrows():
                periods = float(row['periods']) if hasattr(row['periods'], 'iloc') else row['periods']
                scenarios = float(row['scenarios']) if hasattr(row['scenarios'], 'iloc') else row['scenarios']
                
                time_mean = float(row[('total_time', 'mean')]) if hasattr(row[('total_time', 'mean')], 'iloc') else row[('total_time', 'mean')]
                time_std = float(row[('total_time', 'std')]) if hasattr(row[('total_time', 'std')], 'iloc') else row[('total_time', 'std')]
                time_min = float(row[('total_time', 'min')]) if hasattr(row[('total_time', 'min')], 'iloc') else row[('total_time', 'min')]
                time_max = float(row[('total_time', 'max')]) if hasattr(row[('total_time', 'max')], 'iloc') else row[('total_time', 'max')]
                
                iter_mean = float(row[('iterations', 'mean')]) if hasattr(row[('iterations', 'mean')], 'iloc') else row[('iterations', 'mean')]
                obj_mean = float(row[('objective', 'mean')]) if hasattr(row[('objective', 'mean')], 'iloc') else row[('objective', 'mean')]
                
                gap_mean = float(row[('mip_gap', 'mean')]) if hasattr(row[('mip_gap', 'mean')], 'iloc') else row[('mip_gap', 'mean')]
                gap_min = float(row[('mip_gap', 'min')]) if hasattr(row[('mip_gap', 'min')], 'iloc') else row[('mip_gap', 'min')]
                gap_max = float(row[('mip_gap', 'max')]) if hasattr(row[('mip_gap', 'max')], 'iloc') else row[('mip_gap', 'max')]
                
                setup_mean = float(row[('setup_percentage', 'mean')]) if hasattr(row[('setup_percentage', 'mean')], 'iloc') else row[('setup_percentage', 'mean')]
                inv_mean = float(row[('avg_inventory', 'mean')]) if hasattr(row[('avg_inventory', 'mean')], 'iloc') else row[('avg_inventory', 'mean')]
                
                html_report += f"""
                    <tr>
                        <td>{periods:.0f}</td>
                        <td>{scenarios:.0f}</td>
                        <td>{time_mean:.2f}</td>
                        <td>{time_std:.2f}</td>
                        <td>{time_min:.2f}</td>
                        <td>{time_max:.2f}</td>
                        <td>{iter_mean:.1f}</td>
                        <td>{obj_mean:.2f}</td>
                        <td>{gap_mean:.2f}% (min: {gap_min:.2f}%, max: {gap_max:.2f}%)</td>
                        <td>{setup_mean:.1f}</td>
                        <td>{inv_mean:.1f}</td>
                    </tr>
                """
            
            html_report += """
                </table>
                
                <h2>Visualizations</h2>
                
                <div class="chart-container">
                    <h3>Computation Time vs Problem Size</h3>
                    <img src="time_vs_size.png" alt="Computation Time vs Problem Size" style="max-width: 800px;">
                </div>
                
                <div class="chart-container">
                    <h3>Iterations vs Problem Size</h3>
                    <img src="iterations_vs_size.png" alt="Iterations vs Problem Size" style="max-width: 800px;">
                </div>
                
                <div class="chart-container">
                    <h3>Time Breakdown by Problem Component</h3>
                    <img src="time_breakdown.png" alt="Time Breakdown" style="max-width: 800px;">
                </div>
                
                <div class="chart-container">
                    <h3>MIP Gap vs Problem Size</h3>
                    <img src="gap_vs_size.png" alt="MIP Gap vs Problem Size" style="max-width: 800px;">
                </div>
                
                <h2>Raw Data</h2>
                <table>
                    <tr>
                        <th>Periods</th>
                        <th>Scenarios</th>
                        <th>Seed</th>
                        <th>Total Time (s)</th>
                        <th>Iterations</th>
                        <th>Objective Value</th>
                        <th>MIP Gap (%)</th>
                    </tr>
            """
            
            # Add rows for each individual run
            for _, row in success_df.iterrows():
                html_report += f"""
                    <tr>
                        <td>{row['periods']}</td>
                        <td>{row['scenarios']}</td>
                        <td>{row['seed']}</td>
                        <td>{row['total_time']:.2f}</td>
                        <td>{row['iterations']}</td>
                        <td>{row['objective']:.2f}</td>
                        <td>{row['mip_gap']:.2f}</td>
                    </tr>
                """
            
            html_report += """
                </table>
            </body>
            </html>
            """
            
            # Write HTML report
            with open(os.path.join(output_dir, 'performance_report.html'), 'w') as f:
                f.write(html_report)
            
            print(f"\nDetailed report saved to {os.path.join(output_dir, 'performance_report.html')}")
            
        except Exception as e:
            print(f"\nError generating HTML report: {e}")
    
    # Show plots if running in interactive mode
    plt.show()

def main():
    """Main entry point for performance evaluation script."""
    parser = argparse.ArgumentParser(description='Performance evaluation for Benders decomposition')
    parser.add_argument('--problem_sizes', type=str, default="12:2,24:3,36:4", 
                        help='Problem sizes as periods:scenarios,periods:scenarios,...')
    parser.add_argument('--seeds', type=str, default="42,43,44", help='Random seeds')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per problem size')
    parser.add_argument('--use_ml', action='store_true', help='Use ML enhancement')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--max_iterations', type=int, default=30, help='Maximum number of Benders iterations')
    
    args = parser.parse_args()
    
    # Parse problem sizes
    problem_sizes = []
    for size in args.problem_sizes.split(','):
        periods, scenarios = map(int, size.split(':'))
        problem_sizes.append((periods, scenarios))
    
    # Parse seeds
    seeds = list(map(int, args.seeds.split(',')))
    
    print("Starting performance evaluation...")
    print(f"Problem sizes: {problem_sizes}")
    print(f"Seeds: {seeds}")
    print(f"Trials per size: {args.trials}")
    print(f"Using ML: {args.use_ml}")
    
    try:
        # Run benchmarks
        results_df = run_benchmark(
            problem_sizes=problem_sizes,
            seeds=seeds,
            num_trials=args.trials,
            use_ml=args.use_ml,
            output_dir=args.output_dir
        )
        
        # Generate performance report
        generate_performance_report(results_df, args.output_dir)
        
    except Exception as e:
        print(f"Error in performance evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()