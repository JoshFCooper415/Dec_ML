import pickle
import time
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.data_structures import Solution, ProblemParameters
from models.benders import MLBendersDecomposition
from solvers.direct_solver import DirectSolver


def print_solution(solution: Solution, params: ProblemParameters):
    """Print detailed solution information.
    
    Args:
        solution: Solution to print
        params: Problem parameters
    """
    print(f"\nObjective Value: {solution.objective:.2f}")
    
    for s in range(len(params.scenarios)):
        print(f"\nScenario {s} (Probability: {params.scenarios[s].probability}):")
        print("Period | Setup | Production | Inventory | Demand | Linking")
        print("-" * 65)
        
        for t in range(params.total_periods):
            linking_marker = "*" if t in params.linking_periods else " "
            print(f"{t:6d} | {int(solution.setup[s][t]):5d} | "
                  f"{solution.production[s][t]:10.1f} | "
                  f"{solution.inventory[s][t]:9.1f} | "
                  f"{params.scenarios[s].demands[t]:6.1f} | "
                  f"{linking_marker:^7s}")
        
        # Print summary statistics for scenario
        total_setups = sum(1 for x in solution.setup[s] if x)
        avg_inventory = sum(solution.inventory[s]) / len(solution.inventory[s])
        print(f"\nScenario Summary:")
        print(f"Total Setups: {total_setups}")
        print(f"Average Inventory: {avg_inventory:.1f}")


def compare_solutions(solutions: Dict[str, Solution]) -> Dict[str, Dict[str, float]]:
    """Compare multiple solutions.
    
    Args:
        solutions: Dictionary mapping solver name to solution
        
    Returns:
        Dictionary of comparison metrics between each pair of solutions
    """
    comparisons = {}
    solver_names = list(solutions.keys())
    
    # Compare each pair of solutions
    for i in range(len(solver_names)):
        for j in range(i+1, len(solver_names)):
            name_i = solver_names[i]
            name_j = solver_names[j]
            sol_i = solutions[name_i]
            sol_j = solutions[name_j]
            
            # Compare objective values
            obj_diff = abs(sol_i.objective - sol_j.objective)
            obj_rel_diff = obj_diff / sol_j.objective * 100
            
            # Compare setup decisions
            setup_agreement = 0
            setup_total = 0
            
            for s in range(len(sol_i.setup)):
                for t in range(len(sol_i.setup[s])):
                    if sol_i.setup[s][t] == sol_j.setup[s][t]:
                        setup_agreement += 1
                    setup_total += 1
            
            setup_accuracy = setup_agreement / setup_total * 100
            
            # Compare production levels
            prod_mse = 0
            prod_total = 0
            
            for s in range(len(sol_i.production)):
                for t in range(len(sol_i.production[s])):
                    prod_mse += (sol_i.production[s][t] - sol_j.production[s][t]) ** 2
                    prod_total += 1
            
            prod_rmse = (prod_mse / prod_total) ** 0.5
            
            # Store comparison
            pair_key = f"{name_i}_vs_{name_j}"
            comparisons[pair_key] = {
                'obj_diff': obj_diff,
                'obj_rel_diff': obj_rel_diff,
                'setup_accuracy': setup_accuracy,
                'prod_rmse': prod_rmse
            }
            
    return comparisons


def print_comparison_table(comparisons: Dict[str, Dict[str, float]]):
    """Print a table of solution comparisons.
    
    Args:
        comparisons: Dictionary of comparison metrics between solution pairs
    """
    print("\nSolution Comparisons:")
    print("-" * 80)
    print("Comparison | Obj Diff | Obj Rel Diff (%) | Setup Accuracy (%) | Prod RMSE")
    print("-" * 80)
    
    for pair, metrics in comparisons.items():
        print(f"{pair:10s} | {metrics['obj_diff']:8.2f} | "
              f"{metrics['obj_rel_diff']:15.2f} | "
              f"{metrics['setup_accuracy']:17.2f} | "
              f"{metrics['prod_rmse']:9.2f}")


def run_benchmark(problem_sizes=None, seeds=None, num_trials=3, 
                 methods=None, timeout=600):
    """Run benchmarks comparing different solution approaches.
    
    Args:
        problem_sizes: List of tuples (periods, scenarios)
        seeds: List of random seeds
        num_trials: Number of trials to run
        methods: List of methods to benchmark ('direct', 'benders', 'ml_benders')
        timeout: Maximum runtime in seconds
        
    Returns:
        List of benchmark results
    """
    from data.problem_generator import create_test_problem
    
    if problem_sizes is None:
        problem_sizes = [(12, 2), (24, 3), (36, 4)]
    
    if seeds is None:
        seeds = [42, 43, 44]
    
    if methods is None:
        methods = ['direct', 'benders', 'ml_benders']
    
    results = []
    
    for periods, scenarios in problem_sizes:
        for seed in seeds[:num_trials]:
            print(f"\nBenchmarking problem with {periods} periods, {scenarios} scenarios (seed {seed})...")
            
            # Create problem
            params = create_test_problem(periods, scenarios, seed)
            
            # Dictionary to track solutions
            solutions = {}
            timings = {}
            method_status = {}
            
            # Run each method
            for method in methods:
                print(f"Solving with {method}...")
                start_time = time.time()
                
                try:
                    if method == 'direct':
                        # Solve with direct approach
                        direct_solver = DirectSolver(params)
                        obj_value, stats = direct_solver.solve()
                        solutions[method] = direct_solver.best_solution
                        timings[method] = stats.total_solve_time
                        method_status[method] = 'success'
                        
                    elif method == 'benders':
                        # Solve with standard Benders
                        benders = MLBendersDecomposition(params, use_ml=False, use_trust_region=True)
                        lb, ub, stats = benders.solve(max_iterations=30)
                        solutions[method] = benders.best_solution
                        timings[method] = stats.total_solve_time
                        method_status[method] = 'success'
                        
                    elif method == 'ml_benders':
                        # Solve with ML-enhanced Benders
                        ml_benders = MLBendersDecomposition(params, use_ml=True, use_trust_region=True)
                        lb, ub, stats = ml_benders.solve(max_iterations=30)
                        solutions[method] = ml_benders.best_solution
                        timings[method] = stats.total_solve_time
                        method_status[method] = 'success'
                    
                    # Check if we've exceeded timeout
                    if time.time() - start_time > timeout:
                        print(f"  {method} timed out after {timeout} seconds")
                        method_status[method] = 'timeout'
                        
                except Exception as e:
                    print(f"  {method} failed: {str(e)}")
                    method_status[method] = 'failed'
            
            # Compare solutions if we have multiple successful methods
            successful_methods = [m for m in methods if method_status[m] == 'success']
            if len(successful_methods) > 1:
                successful_solutions = {m: solutions[m] for m in successful_methods}
                comparisons = compare_solutions(successful_solutions)
                print_comparison_table(comparisons)
            
            # Calculate speedups relative to direct solver (if available)
            speedups = {}
            if 'direct' in timings:
                direct_time = timings['direct']
                for method in methods:
                    if method != 'direct' and method in timings:
                        speedups[f"{method}_vs_direct"] = direct_time / timings[method]
            
            # Store result
            result = {
                'periods': periods,
                'scenarios': scenarios,
                'seed': seed,
                'status': method_status,
                'timings': timings,
                'speedups': speedups
            }
            
            # Add objectives if available
            for method in methods:
                if method in solutions:
                    result[f'{method}_objective'] = solutions[method].objective
            
            # Add comparisons if available
            if len(successful_methods) > 1:
                for pair, metrics in comparisons.items():
                    for metric, value in metrics.items():
                        result[f'{pair}_{metric}'] = value
            
            results.append(result)
    
    # Print summary table
    print("\nBenchmark Summary:")
    print("-" * 120)
    print("Size | Direct | Benders | ML-Benders | Direct/Benders | Direct/ML | ML/Benders | Obj Diff %")
    print("-" * 120)
    
    for result in results:
        periods = result['periods']
        scenarios = result['scenarios']
        size_str = f"{periods}p,{scenarios}s"
        
        # Get timings or placeholders
        direct_time = f"{result['timings'].get('direct', 0):.2f}s" if 'direct' in result['timings'] else "N/A"
        benders_time = f"{result['timings'].get('benders', 0):.2f}s" if 'benders' in result['timings'] else "N/A"
        ml_time = f"{result['timings'].get('ml_benders', 0):.2f}s" if 'ml_benders' in result['timings'] else "N/A"
        
        # Get speedups or placeholders
        d_b_speedup = f"{result['speedups'].get('benders_vs_direct', 0):.2f}x" if 'benders_vs_direct' in result['speedups'] else "N/A"
        d_ml_speedup = f"{result['speedups'].get('ml_benders_vs_direct', 0):.2f}x" if 'ml_benders_vs_direct' in result['speedups'] else "N/A"
        
        # Calculate ML vs Benders speedup
        ml_b_speedup = "N/A"
        if 'ml_benders' in result['timings'] and 'benders' in result['timings']:
            if result['timings']['benders'] > 0:
                speedup = result['timings']['benders'] / result['timings']['ml_benders']
                ml_b_speedup = f"{speedup:.2f}x"
        
        # Get objective difference or placeholder
        obj_diff = "N/A"
        if 'ml_benders_vs_benders_obj_rel_diff' in result:
            obj_diff = f"{result['ml_benders_vs_benders_obj_rel_diff']:.2f}%"
        
        print(f"{size_str:8s} | {direct_time:7s} | {benders_time:8s} | {ml_time:11s} | "
              f"{d_b_speedup:14s} | {d_ml_speedup:10s} | {ml_b_speedup:11s} | {obj_diff:9s}")
    
    return results


def plot_results(results, output_file="benchmark_plot.png"):
    """Plot benchmark results with more robust error handling.
    
    Args:
        results: List of benchmark results
        output_file: Output file name for plot or None to display
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if we have any successful results to plot
    any_success = False
    for result in results:
        for method in result.get('status', {}):
            if result['status'].get(method) == 'success':
                any_success = True
                break
        if any_success:
            break
    
    if not any_success:
        print("No successful benchmark runs to plot. Skipping plot generation.")
        return
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Extract problem sizes
    df['problem_size'] = df.apply(lambda x: f"{x['periods']}p,{x['scenarios']}s", axis=1)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot 1: Solution times
    problem_sizes = sorted(df['problem_size'].unique())
    methods = ['direct', 'benders', 'ml_benders']
    method_labels = ['Direct', 'Benders', 'ML-Benders']
    colors = ['blue', 'orange', 'green']
    
    # Group by problem size and calculate average times
    times_data = []
    for size in problem_sizes:
        size_df = df[df['problem_size'] == size]
        
        for method in methods:
            # Extract timings for successful runs
            method_times = []
            for idx, row in size_df.iterrows():
                if method in row.get('timings', {}) and row.get('status', {}).get(method) == 'success':
                    method_times.append(row['timings'][method])
            
            # Calculate average if we have data
            if method_times:
                avg_time = sum(method_times) / len(method_times)
                times_data.append({'problem_size': size, 'method': method, 'time': avg_time})
    
    # Check if we have timing data to plot
    if not times_data:
        print("No timing data available to plot.")
        fig.text(0.5, 0.5, 'No successful benchmark runs to plot', 
                ha='center', va='center', fontsize=20)
        plt.tight_layout()
        plt.savefig(output_file)
        return
    
    # Convert to DataFrame
    times_df = pd.DataFrame(times_data)
    
    # Plot grouped bar chart of solution times
    width = 0.2
    x = np.arange(len(problem_sizes))
    
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        method_times = []
        for size in problem_sizes:
            method_size_df = times_df[(times_df['problem_size'] == size) & (times_df['method'] == method)]
            if len(method_size_df) > 0:
                method_times.append(method_size_df['time'].values[0])
            else:
                method_times.append(0)
        
        axs[0].bar(x + (i - 1) * width, method_times, width, label=label, color=colors[i])
    
    axs[0].set_xlabel('Problem Size')
    axs[0].set_ylabel('Avg. Solution Time (s)')
    axs[0].set_title('Solution Times by Method and Problem Size')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(problem_sizes)
    axs[0].legend()
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Subplot 2: Speedups
    speedup_data = []
    for size in problem_sizes:
        size_df = df[df['problem_size'] == size]
        
        # Calculate average speedups
        d_b_speedups = []
        d_ml_speedups = []
        ml_b_speedups = []
        
        for idx, row in size_df.iterrows():
            if 'benders_vs_direct' in row.get('speedups', {}):
                d_b_speedups.append(row['speedups']['benders_vs_direct'])
            
            if 'ml_benders_vs_direct' in row.get('speedups', {}):
                d_ml_speedups.append(row['speedups']['ml_benders_vs_direct'])
            
            # Calculate ML vs Benders speedup
            if 'ml_benders' in row.get('timings', {}) and 'benders' in row.get('timings', {}):
                if row.get('status', {}).get('ml_benders') == 'success' and row.get('status', {}).get('benders') == 'success':
                    if row['timings']['benders'] > 0:
                        speedup = row['timings']['benders'] / row['timings']['ml_benders']
                        ml_b_speedups.append(speedup)
        
        # Add average speedups to data
        if d_b_speedups:
            speedup_data.append({'problem_size': size, 'comparison': 'Direct/Benders', 
                                'speedup': sum(d_b_speedups) / len(d_b_speedups)})
        
        if d_ml_speedups:
            speedup_data.append({'problem_size': size, 'comparison': 'Direct/ML-Benders', 
                                'speedup': sum(d_ml_speedups) / len(d_ml_speedups)})
        
        if ml_b_speedups:
            speedup_data.append({'problem_size': size, 'comparison': 'ML-Benders/Benders', 
                                'speedup': sum(ml_b_speedups) / len(ml_b_speedups)})
    
    # Convert to DataFrame if we have data
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        
        # Plot grouped bar chart of speedups
        comparisons = ['Direct/Benders', 'Direct/ML-Benders', 'ML-Benders/Benders']
        colors = ['purple', 'red', 'teal']
        
        for i, (comparison, color) in enumerate(zip(comparisons, colors)):
            comp_speedups = []
            for size in problem_sizes:
                comp_size_df = speedup_df[(speedup_df['problem_size'] == size) & 
                                        (speedup_df['comparison'] == comparison)]
                if len(comp_size_df) > 0:
                    comp_speedups.append(comp_size_df['speedup'].values[0])
                else:
                    comp_speedups.append(0)
            
            axs[1].bar(x + (i - 1) * width, comp_speedups, width, label=comparison, color=color)
        
        axs[1].set_xlabel('Problem Size')
        axs[1].set_ylabel('Average Speedup (x)')
        axs[1].set_title('Speedups by Method Comparison and Problem Size')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(problem_sizes)
        axs[1].legend()
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        axs[1].text(0.5, 0.5, 'No speedup data available', 
                   ha='center', va='center', fontsize=14, transform=axs[1].transAxes)
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    except Exception as e:
        print(f"Failed to save plot: {str(e)}")
        

def save_benchmark_results(results: List[Dict[str, Any]], filename: str = "benchmark_results.pkl"):
    """Save benchmark results to a file.
    
    Args:
        results: List of benchmark results
        filename: Output file name
    """
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nBenchmark results saved to {filename}")


if __name__ == "__main__":
    # Run benchmarks with default settings
    results = run_benchmark()
    
    # Save results
    save_benchmark_results(results)
    
    # Plot results
    plot_results(results)