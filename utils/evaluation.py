import pickle
from typing import Dict, List, Any

from ..data.data_structures import Solution, ProblemParameters

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

def compare_solutions(ml_solution: Solution, exact_solution: Solution) -> Dict[str, float]:
    """Compare ML-based solution with exact solution.
    
    Args:
        ml_solution: Solution from ML-enhanced solver
        exact_solution: Solution from exact solver
        
    Returns:
        Dictionary of comparison metrics
    """
    # Compare objective values
    obj_diff = abs(ml_solution.objective - exact_solution.objective)
    obj_rel_diff = obj_diff / exact_solution.objective * 100
    
    # Compare setup decisions
    setup_agreement = 0
    setup_total = 0
    
    for s in range(len(ml_solution.setup)):
        for t in range(len(ml_solution.setup[s])):
            if ml_solution.setup[s][t] == exact_solution.setup[s][t]:
                setup_agreement += 1
            setup_total += 1
    
    setup_accuracy = setup_agreement / setup_total * 100
    
    # Compare production levels
    prod_mse = 0
    prod_total = 0
    
    for s in range(len(ml_solution.production)):
        for t in range(len(ml_solution.production[s])):
            prod_mse += (ml_solution.production[s][t] - exact_solution.production[s][t]) ** 2
            prod_total += 1
    
    prod_rmse = (prod_mse / prod_total) ** 0.5
    
    # Print comparison
    print("\nSolution Comparison:")
    print(f"Objective Value - ML: {ml_solution.objective:.2f}, Exact: {exact_solution.objective:.2f}")
    print(f"Objective Difference: {obj_diff:.2f} ({obj_rel_diff:.2f}%)")
    print(f"Setup Decision Accuracy: {setup_accuracy:.2f}%")
    print(f"Production RMSE: {prod_rmse:.2f}")
    
    return {
        'obj_diff': obj_diff,
        'obj_rel_diff': obj_rel_diff,
        'setup_accuracy': setup_accuracy,
        'prod_rmse': prod_rmse
    }

def run_benchmark(problem_sizes=None, seeds=None, num_trials=3):
    """Run benchmarks comparing ML-based approach with exact solution.
    
    Args:
        problem_sizes: List of tuples (periods, scenarios)
        seeds: List of random seeds
        num_trials: Number of trials to run
        
    Returns:
        List of benchmark results
    """
    from ..data.problem_generator import create_test_problem
    from ..models.benders import MLBendersDecomposition
    
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
            
            # Solve with ML
            print("Solving with ML...")
            ml_decomp = MLBendersDecomposition(params, use_ml=True)
            try:
                ml_lb, ml_ub, ml_stats = ml_decomp.solve(max_iterations=30)
                ml_success = True
            except Exception as e:
                print(f"ML solution failed: {str(e)}")
                ml_success = False
            
            # Solve without ML (exact)
            print("Solving without ML (exact)...")
            exact_decomp = MLBendersDecomposition(params, use_ml=False)
            try:
                exact_lb, exact_ub, exact_stats = exact_decomp.solve(max_iterations=30)
                exact_success = True
            except Exception as e:
                print(f"Exact solution failed: {str(e)}")
                exact_success = False
            
            # Compare if both succeeded
            if ml_success and exact_success:
                comparison = compare_solutions(ml_decomp.best_solution, exact_decomp.best_solution)
                speedup = exact_stats.total_solve_time / ml_stats.total_solve_time
                
                result = {
                    'periods': periods,
                    'scenarios': scenarios,
                    'seed': seed,
                    'ml_objective': ml_ub,
                    'exact_objective': exact_ub,
                    'ml_time': ml_stats.total_solve_time,
                    'exact_time': exact_stats.total_solve_time,
                    'ml_iterations': ml_stats.num_iterations,
                    'exact_iterations': exact_stats.num_iterations,
                    'ml_prediction_time': ml_stats.ml_prediction_time,
                    'speedup': speedup,
                    'obj_rel_diff': comparison['obj_rel_diff'],
                    'setup_accuracy': comparison['setup_accuracy'],
                    'prod_rmse': comparison['prod_rmse'],
                    'status': 'success'
                }
                
                # Print timing comparison
                print("\nTiming Comparison:")
                print(f"ML Total Time: {ml_stats.total_solve_time:.2f}s (ML Prediction: {ml_stats.ml_prediction_time:.2f}s)")
                print(f"Exact Total Time: {exact_stats.total_solve_time:.2f}s")
                print(f"Speedup: {speedup:.2f}x")
                
            elif ml_success:
                result = {
                    'periods': periods,
                    'scenarios': scenarios,
                    'seed': seed,
                    'ml_objective': ml_ub,
                    'ml_time': ml_stats.total_solve_time,
                    'ml_iterations': ml_stats.num_iterations,
                    'status': 'ml_only'
                }
            elif exact_success:
                result = {
                    'periods': periods,
                    'scenarios': scenarios,
                    'seed': seed,
                    'exact_objective': exact_ub,
                    'exact_time': exact_stats.total_solve_time,
                    'exact_iterations': exact_stats.num_iterations,
                    'status': 'exact_only'
                }
            else:
                result = {
                    'periods': periods,
                    'scenarios': scenarios,
                    'seed': seed,
                    'status': 'both_failed'
                }
            
            results.append(result)
    
    # Print summary table
    print("\nBenchmark Summary:")
    print("-" * 100)
    print("Periods | Scenarios | Exact Time | ML Time | Speedup | Obj Diff % | Setup Acc % | Status")
    print("-" * 100)
    
    for result in results:
        periods = result['periods']
        scenarios = result['scenarios']
        
        if result['status'] == 'success':
            print(f"{periods:7d} | {scenarios:9d} | {result['exact_time']:10.2f}s | "
                  f"{result['ml_time']:7.2f}s | {result['speedup']:7.2f}x | "
                  f"{result['obj_rel_diff']:9.2f}% | {result['setup_accuracy']:10.2f}% | {result['status']}")
        else:
            print(f"{periods:7d} | {scenarios:9d} | {'---':>10s} | {'---':>7s} | {'---':>7s} | "
                  f"{'---':>9s} | {'---':>10s} | {result['status']}")
    
    return results

def save_benchmark_results(results: List[Dict[str, Any]], filename: str = "benchmark_results.pkl"):
    """Save benchmark results to a file.
    
    Args:
        results: List of benchmark results
        filename: Output file name
    """
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nBenchmark results saved to {filename}")