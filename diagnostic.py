#!/usr/bin/env python
"""
Comprehensive diagnostic test for Benders decomposition in production planning.

This script performs detailed comparisons between direct solver and Benders decomposition
to identify sources of discrepancy in objective values. It includes cross-validation
by evaluating each solver's solution within the other solver's framework.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
import gurobipy as gp
from gurobipy import GRB

# Add current directory to path to help with imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import required modules
from data.data_structures import ScenarioData, ProblemParameters, Solution, TimingStats
from data.problem_generator import create_test_problem
from models.benders import MLBendersDecomposition
from solvers.direct_solver import DirectSolver


def calculate_objective_manually(sol: Solution, params: ProblemParameters) -> float:
    """Calculate objective value manually to verify solver calculations."""
    total_obj = 0.0
    
    for s_idx, scenario in enumerate(params.scenarios):
        scenario_obj = 0.0
        
        # Sum setup costs
        for t in range(len(sol.setup[s_idx])):
            if sol.setup[s_idx][t]:
                scenario_obj += params.fixed_cost
        
        # Sum production costs
        for t in range(len(sol.production[s_idx])):
            scenario_obj += params.production_cost * sol.production[s_idx][t]
        
        # Sum inventory holding costs
        for t in range(len(sol.inventory[s_idx])):
            scenario_obj += params.holding_cost * sol.inventory[s_idx][t]
        
        # Add weighted scenario objective
        total_obj += scenario.probability * scenario_obj
    
    return total_obj


def compare_solutions(direct_sol: Solution, benders_sol: Solution) -> Dict[str, Any]:
    """Compare direct and Benders solutions in detail."""
    # Compare objective values
    direct_obj = direct_sol.objective
    benders_obj = benders_sol.objective
    obj_gap = abs(direct_obj - benders_obj) / max(1e-10, abs(direct_obj)) * 100
    
    # Compare setup decisions
    setup_matches = 0
    setup_total = 0
    setup_diffs = []  # Track which periods have different setup decisions
    
    for s in range(len(direct_sol.setup)):
        for t in range(len(direct_sol.setup[s])):
            if direct_sol.setup[s][t] == benders_sol.setup[s][t]:
                setup_matches += 1
            else:
                setup_diffs.append((s, t))
            setup_total += 1
            
    setup_match_pct = (setup_matches / max(1, setup_total)) * 100
    
    return {
        'objective_gap_pct': obj_gap,
        'setup_match_pct': setup_match_pct,
        'setup_diffs': setup_diffs
    }


def analyze_objective_components(params: ProblemParameters, 
                              direct_sol: Solution, 
                              benders_sol: Solution) -> Dict[str, Any]:
    """Analyze objective components to find where discrepancies occur."""
    # Calculate component costs for each solution
    direct_setup_cost = 0.0
    direct_prod_cost = 0.0
    direct_hold_cost = 0.0
    
    benders_setup_cost = 0.0
    benders_prod_cost = 0.0
    benders_hold_cost = 0.0
    
    # Detailed analysis by scenario and period
    scenario_details = []
    
    for s_idx, scenario in enumerate(params.scenarios):
        prob = scenario.probability
        s_detail = {'scenario': s_idx, 'probability': prob, 'periods': []}
        
        # Calculate costs by period
        for t in range(min(len(direct_sol.setup[s_idx]), len(benders_sol.setup[s_idx]))):
            # Direct solution
            direct_setup = 1 if direct_sol.setup[s_idx][t] else 0
            direct_setup_cost_t = direct_setup * params.fixed_cost
            
            direct_prod = direct_sol.production[s_idx][t] if t < len(direct_sol.production[s_idx]) else 0
            direct_prod_cost_t = direct_prod * params.production_cost
            
            direct_inv = direct_sol.inventory[s_idx][t] if t < len(direct_sol.inventory[s_idx]) else 0
            direct_hold_cost_t = direct_inv * params.holding_cost
            
            # Benders solution
            benders_setup = 1 if benders_sol.setup[s_idx][t] else 0
            benders_setup_cost_t = benders_setup * params.fixed_cost
            
            benders_prod = benders_sol.production[s_idx][t] if t < len(benders_sol.production[s_idx]) else 0
            benders_prod_cost_t = benders_prod * params.production_cost
            
            benders_inv = benders_sol.inventory[s_idx][t] if t < len(benders_sol.inventory[s_idx]) else 0
            benders_hold_cost_t = benders_inv * params.holding_cost
            
            # Add to totals (weighted by probability)
            direct_setup_cost += prob * direct_setup_cost_t
            direct_prod_cost += prob * direct_prod_cost_t
            direct_hold_cost += prob * direct_hold_cost_t
            
            benders_setup_cost += prob * benders_setup_cost_t
            benders_prod_cost += prob * benders_prod_cost_t
            benders_hold_cost += prob * benders_hold_cost_t
            
            # Record period details
            p_detail = {
                'period': t,
                'is_linking': t in params.linking_periods,
                'direct': {
                    'setup': direct_setup,
                    'production': direct_prod,
                    'inventory': direct_inv,
                    'setup_cost': direct_setup_cost_t,
                    'prod_cost': direct_prod_cost_t,
                    'hold_cost': direct_hold_cost_t,
                    'total': direct_setup_cost_t + direct_prod_cost_t + direct_hold_cost_t
                },
                'benders': {
                    'setup': benders_setup,
                    'production': benders_prod,
                    'inventory': benders_inv,
                    'setup_cost': benders_setup_cost_t,
                    'prod_cost': benders_prod_cost_t,
                    'hold_cost': benders_hold_cost_t,
                    'total': benders_setup_cost_t + benders_prod_cost_t + benders_hold_cost_t
                },
                'diff': {
                    'setup': direct_setup - benders_setup,
                    'production': direct_prod - benders_prod,
                    'inventory': direct_inv - benders_inv,
                    'setup_cost': direct_setup_cost_t - benders_setup_cost_t,
                    'prod_cost': direct_prod_cost_t - benders_prod_cost_t,
                    'hold_cost': direct_hold_cost_t - benders_hold_cost_t,
                    'total': (direct_setup_cost_t + direct_prod_cost_t + direct_hold_cost_t) - 
                            (benders_setup_cost_t + benders_prod_cost_t + benders_hold_cost_t)
                }
            }
            s_detail['periods'].append(p_detail)
        
        scenario_details.append(s_detail)
    
    direct_total = direct_setup_cost + direct_prod_cost + direct_hold_cost
    benders_total = benders_setup_cost + benders_prod_cost + benders_hold_cost
    
    # Calculate component differences
    setup_diff = benders_setup_cost - direct_setup_cost
    setup_diff_pct = (setup_diff / max(1e-10, abs(direct_setup_cost))) * 100
    
    prod_diff = benders_prod_cost - direct_prod_cost
    prod_diff_pct = (prod_diff / max(1e-10, abs(direct_prod_cost))) * 100
    
    hold_diff = benders_hold_cost - direct_hold_cost
    hold_diff_pct = (hold_diff / max(1e-10, abs(direct_hold_cost))) * 100
    
    total_diff = benders_total - direct_total
    total_diff_pct = (total_diff / max(1e-10, abs(direct_total))) * 100
    
    print("\n=== Objective Component Analysis ===")
    print(f"Direct solver cost components:")
    print(f"  Setup cost:     {direct_setup_cost:.2f} ({direct_setup_cost/direct_total*100:.1f}%)")
    print(f"  Production cost: {direct_prod_cost:.2f} ({direct_prod_cost/direct_total*100:.1f}%)")
    print(f"  Holding cost:   {direct_hold_cost:.2f} ({direct_hold_cost/direct_total*100:.1f}%)")
    print(f"  Total cost:     {direct_total:.2f}")
    
    print(f"\nBenders solver cost components:")
    print(f"  Setup cost:     {benders_setup_cost:.2f} ({benders_setup_cost/benders_total*100:.1f}%)")
    print(f"  Production cost: {benders_prod_cost:.2f} ({benders_prod_cost/benders_total*100:.1f}%)")
    print(f"  Holding cost:   {benders_hold_cost:.2f} ({benders_hold_cost/benders_total*100:.1f}%)")
    print(f"  Total cost:     {benders_total:.2f}")
    
    print(f"\nCost differences (Benders - Direct):")
    print(f"  Setup cost diff:     {setup_diff:.2f} ({setup_diff_pct:.4f}%)")
    print(f"  Production cost diff: {prod_diff:.2f} ({prod_diff_pct:.4f}%)")
    print(f"  Holding cost diff:   {hold_diff:.2f} ({hold_diff_pct:.4f}%)")
    print(f"  Total cost diff:     {total_diff:.2f} ({total_diff_pct:.4f}%)")
    
    return {
        'component_summary': {
            'direct_setup_cost': direct_setup_cost,
            'direct_prod_cost': direct_prod_cost,
            'direct_hold_cost': direct_hold_cost,
            'direct_total': direct_total,
            'benders_setup_cost': benders_setup_cost,
            'benders_prod_cost': benders_prod_cost,
            'benders_hold_cost': benders_hold_cost,
            'benders_total': benders_total,
            'setup_diff': setup_diff,
            'setup_diff_pct': setup_diff_pct,
            'prod_diff': prod_diff,
            'prod_diff_pct': prod_diff_pct,
            'hold_diff': hold_diff,
            'hold_diff_pct': hold_diff_pct,
            'total_diff': total_diff,
            'total_diff_pct': total_diff_pct
        },
        'scenario_details': scenario_details
    }


def identify_key_differences(analysis_results: Dict) -> None:
    """Identify and report key differences between solutions."""
    print("\n=== Key Differences Analysis ===")
    
    # Extract scenario details
    scenarios = analysis_results['scenario_details']
    
    # Find periods with significant differences
    sig_diff_periods = []
    
    for s_idx, scenario in enumerate(scenarios):
        for period in scenario['periods']:
            t = period['period']
            # Check if difference is significant (>0.1% of total cost)
            if abs(period['diff']['total']) > 0.001 * analysis_results['component_summary']['direct_total']:
                sig_diff_periods.append((s_idx, t, period['diff']['total']))
    
    # Sort by absolute difference
    sig_diff_periods.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if sig_diff_periods:
        print(f"Top significant differences (scenario, period, difference):")
        for i, (s, t, diff) in enumerate(sig_diff_periods[:10]):  # Show top 10
            print(f"  {i+1}. Scenario {s}, Period {t}: {diff:.2f}")
    else:
        print("No significant differences found at period level.")
    
    # Analyze setup differences specifically
    setup_diffs = []
    
    for s_idx, scenario in enumerate(scenarios):
        for period in scenario['periods']:
            t = period['period']
            if period['diff']['setup'] != 0:
                # This is a period where setup decisions differ
                is_linking = period['is_linking']
                setup_diffs.append((s_idx, t, is_linking))
    
    if setup_diffs:
        print(f"\nSetup decision differences (scenario, period, is linking period):")
        for i, (s, t, is_linking) in enumerate(setup_diffs):
            link_str = "LINKING" if is_linking else "regular"
            print(f"  {i+1}. Scenario {s}, Period {t}: {link_str}")
    else:
        print("\nNo setup decision differences found.")


def cross_validate_solutions(params: ProblemParameters, direct_sol: Solution, benders_sol: Solution):
    """
    Cross-validate solutions between Direct and Benders solvers.
    Evaluates Benders solution in Direct solver and vice versa.
    """
    print("\n=== Cross-Validation of Solutions ===")
    
    # 1. Extract setup decisions
    direct_setups_by_scenario = {}
    benders_setups_by_scenario = {}
    
    for s in range(len(params.scenarios)):
        direct_setups_by_scenario[s] = {}
        benders_setups_by_scenario[s] = {}
        for t in range(params.total_periods):
            direct_setups_by_scenario[s][t] = direct_sol.setup[s][t]
            benders_setups_by_scenario[s][t] = benders_sol.setup[s][t]
    
    # 2. Evaluate Benders solution in Direct solver
    print("\nEvaluating Benders solution in Direct solver...")
    direct_with_benders = DirectSolver(params)
    direct_model = direct_with_benders.build_model()
    
    # Fix setup variables to match Benders solution
    for s in range(len(params.scenarios)):
        for t in range(params.total_periods):
            var_name = f"setup_{s}_{t}"
            var = direct_model.getVarByName(var_name)
            if var:
                if benders_sol.setup[s][t]:
                    var.lb = 1.0
                else:
                    var.ub = 0.0
    
    # Also fix the here-and-now variables for linking periods
    for t in params.linking_periods:
        here_setup_var = direct_model.getVarByName(f"here_setup_{t}")
        if here_setup_var:
            if benders_sol.setup[0][t]:  # Use first scenario for linking periods
                here_setup_var.lb = 1.0
            else:
                here_setup_var.ub = 0.0
    
    direct_model.optimize()
    
    if direct_model.status == GRB.OPTIMAL:
        benders_in_direct_obj = direct_model.objVal
        print(f"  Objective value: {benders_in_direct_obj:.2f}")
        
        # Calculate component costs
        setup_cost = 0.0
        prod_cost = 0.0
        hold_cost = 0.0
        
        for s in range(len(params.scenarios)):
            prob = params.scenarios[s].probability
            s_setup_cost = 0.0
            s_prod_cost = 0.0
            s_hold_cost = 0.0
            
            for t in range(params.total_periods):
                setup_var = direct_model.getVarByName(f"setup_{s}_{t}")
                prod_var = direct_model.getVarByName(f"production_{s}_{t}")
                inv_var = direct_model.getVarByName(f"inventory_{s}_{t}")
                
                if setup_var and setup_var.x > 0.5:
                    s_setup_cost += params.fixed_cost
                
                if prod_var:
                    s_prod_cost += params.production_cost * prod_var.x
                
                if inv_var:
                    s_hold_cost += params.holding_cost * inv_var.x
            
            setup_cost += prob * s_setup_cost
            prod_cost += prob * s_prod_cost
            hold_cost += prob * s_hold_cost
        
        print(f"  Component costs:")
        print(f"    Setup: {setup_cost:.2f}")
        print(f"    Production: {prod_cost:.2f}")
        print(f"    Holding: {hold_cost:.2f}")
        print(f"    Total: {setup_cost + prod_cost + hold_cost:.2f}")
    else:
        print(f"  Solver status: {direct_model.status} - Could not find optimal solution!")
        benders_in_direct_obj = float('inf')
    
    # 3. Evaluate Direct solution in Benders framework
    print("\nEvaluating Direct solution in Benders framework...")
    
    # We'll need to manually solve each scenario subproblem with direct setups
    direct_in_benders_obj = 0.0
    setup_cost = 0.0
    prod_cost = 0.0
    hold_cost = 0.0
    
    for s_idx, scenario in enumerate(params.scenarios):
        # Setup dictionary for fixed variables
        fixed_setups = {t: direct_sol.setup[s_idx][t] for t in range(params.total_periods)}
        
        # Create and solve a single scenario subproblem
        print(f"  Solving scenario {s_idx}...")
        benders = MLBendersDecomposition(params, use_ml=False)
        scenario_obj, scenario_sol = benders.scenario_problems[s_idx].solve_recursive(fixed_setups)
        
        if scenario_obj < float('inf'):
            # Calculate cost components for this scenario
            s_setup_cost = sum(params.fixed_cost for t in range(len(scenario_sol['setup'])) if scenario_sol['setup'][t])
            s_prod_cost = sum(params.production_cost * scenario_sol['production'][t] for t in range(len(scenario_sol['production'])))
            s_hold_cost = sum(params.holding_cost * scenario_sol['inventory'][t] for t in range(len(scenario_sol['inventory'])))
            
            # Add to weighted totals
            prob = scenario.probability
            setup_cost += prob * s_setup_cost
            prod_cost += prob * s_prod_cost
            hold_cost += prob * s_hold_cost
            direct_in_benders_obj += prob * scenario_obj
            
            print(f"    Objective: {scenario_obj:.2f}")
            print(f"    Setup cost: {s_setup_cost:.2f}")
            print(f"    Production cost: {s_prod_cost:.2f}")
            print(f"    Holding cost: {s_hold_cost:.2f}")
        else:
            print(f"    Infeasible!")
            direct_in_benders_obj = float('inf')
            break
    
    if direct_in_benders_obj < float('inf'):
        print(f"  Total objective: {direct_in_benders_obj:.2f}")
        print(f"  Component costs:")
        print(f"    Setup: {setup_cost:.2f}")
        print(f"    Production: {prod_cost:.2f}")
        print(f"    Holding: {hold_cost:.2f}")
        print(f"    Total: {setup_cost + prod_cost + hold_cost:.2f}")
    
    # 4. Report cross-validation results
    print("\nCross-Validation Results Summary:")
    print(f"Original Direct objective:             {direct_sol.objective:.2f}")
    print(f"Original Benders objective:            {benders_sol.objective:.2f}")
    print(f"Benders solution in Direct solver:     {benders_in_direct_obj:.2f}")
    print(f"Direct solution in Benders framework:  {direct_in_benders_obj:.2f}")
    
    # Calculate errors when feasible
    if benders_in_direct_obj < float('inf') and direct_in_benders_obj < float('inf'):
        # Formula 1: |Direct(Benders) - Benders| / Benders
        benders_discrepancy = abs(benders_in_direct_obj - benders_sol.objective) / benders_sol.objective * 100
        
        # Formula 2: |Benders(Direct) - Direct| / Direct
        direct_discrepancy = abs(direct_in_benders_obj - direct_sol.objective) / direct_sol.objective * 100
        
        print(f"\nDiscrepancy metrics:")
        print(f"  Benders solution evaluated in Direct: {benders_discrepancy:.4f}%")
        print(f"  Direct solution evaluated in Benders: {direct_discrepancy:.4f}%")
        
        if benders_discrepancy > 1.0 or direct_discrepancy > 1.0:
            print("\nSIGNIFICANT DISCREPANCY DETECTED! The solvers are likely solving different formulations.")
        else:
            print("\nThe discrepancy is minimal. The formulations are likely equivalent.")
    else:
        if benders_in_direct_obj == float('inf'):
            print("\nERROR: Benders solution is infeasible in Direct solver!")
        if direct_in_benders_obj == float('inf'):
            print("\nERROR: Direct solution is infeasible in Benders framework!")
        print("\nCritical issue: Solutions are not compatible across solvers. Formulations are definitely different.")
    
    return {
        'direct_obj': direct_sol.objective,
        'benders_obj': benders_sol.objective,
        'benders_in_direct_obj': benders_in_direct_obj,
        'direct_in_benders_obj': direct_in_benders_obj
    }


def check_formulation_parameters(params, direct_solver, benders):
    """Compare parameter usage between solvers."""
    print("\n=== Formulation Parameter Comparison ===")
    
    # Print basic parameter values
    print(f"Fixed cost:       {params.fixed_cost}")
    print(f"Production cost:  {params.production_cost}")
    print(f"Holding cost:     {params.holding_cost}")
    print(f"Capacity:         {params.capacity}")
    print(f"Total periods:    {params.total_periods}")
    print(f"Scenarios:        {len(params.scenarios)}")
    
    # Compare linking periods
    print(f"Linking periods:  {sorted(params.linking_periods)}")
    
    # Check scenario probabilities
    print("\nScenario probabilities:")
    prob_sum = 0
    for i, scenario in enumerate(params.scenarios):
        print(f"  Scenario {i}: {scenario.probability}")
        prob_sum += scenario.probability
    print(f"  Sum: {prob_sum}")
    
    # Check if both solvers are using the same here-and-now variable naming
    print("\nVariable naming:")
    print(f"  Direct solver here-and-now: here_setup_t")
    print(f"  Benders master here-and-now: setup[t]")


def run_direct_vs_benders_test(params: ProblemParameters) -> Dict[str, Any]:
    """Run comprehensive diagnostic comparison between direct and Benders solvers."""
    print("\n=== Running Direct vs. Benders Comparison ===")
    
    # 1. Solve with direct solver
    print("Solving with direct solver...")
    direct_solver = DirectSolver(params)
    direct_start = time.time()
    direct_obj, direct_stats = direct_solver.solve()
    direct_time = time.time() - direct_start
    direct_sol = direct_solver.best_solution
    
    # Calculate objective manually to verify
    direct_obj_manual = calculate_objective_manually(direct_sol, params)
    direct_obj_diff_pct = abs(direct_obj - direct_obj_manual) / max(1e-10, abs(direct_obj)) * 100
    
    # 2. Solve with Benders decomposition
    print("Solving with Benders decomposition...")
    benders = MLBendersDecomposition(params, use_ml=False, use_trust_region=True, use_valid_cuts=True)
    benders_start = time.time()
    benders_lb, benders_ub, benders_stats = benders.solve(max_iterations=30)
    benders_time = time.time() - benders_start
    benders_sol = benders.best_solution
    
    # Calculate objective manually to verify
    benders_obj_manual = calculate_objective_manually(benders_sol, params)
    benders_obj_diff_pct = abs(benders_ub - benders_obj_manual) / max(1e-10, abs(benders_ub)) * 100
    
    # 3. Compare solutions
    solution_comparison = compare_solutions(direct_sol, benders_sol)
    
    # 4. Print summary results
    print("\n=== Test Results Summary ===")
    print(f"Direct solver objective:        {direct_obj:.2f}")
    print(f"Direct solver manual objective: {direct_obj_manual:.2f} (diff: {direct_obj_diff_pct:.8f}%)")
    print(f"Direct solver time:             {direct_time:.2f} seconds")
    
    print(f"\nBenders solver objective:      {benders_ub:.2f}")
    print(f"Benders solver lower bound:     {benders_lb:.2f}")
    print(f"Benders solver MIP gap:         {(benders_ub - benders_lb) / max(1e-10, abs(benders_ub)) * 100:.8f}%")
    print(f"Benders solver manual objective: {benders_obj_manual:.2f} (diff: {benders_obj_diff_pct:.8f}%)")
    print(f"Benders solver time:            {benders_time:.2f} seconds")
    print(f"Benders solver iterations:      {benders_stats.num_iterations}")
    
    print(f"\nDirect vs Benders comparison:")
    print(f"Objective gap:                  {solution_comparison['objective_gap_pct']:.8f}%")
    print(f"Setup match percentage:         {solution_comparison['setup_match_pct']:.2f}%")
    
    # Print linking period setups for comparison
    direct_linking_setups = {t: direct_sol.setup[0][t] for t in params.linking_periods}
    benders_linking_setups = {t: benders_sol.setup[0][t] for t in params.linking_periods}
    print(f"Direct solver linking period setups: {direct_linking_setups}")
    print(f"Benders solver linking period setups: {benders_linking_setups}")
    
    # 5. Detailed analysis if gap exists
    if solution_comparison['objective_gap_pct'] > 0.01:
        print("\nDetected significant gap between Direct and Benders solutions.")
        
        # Check formulation parameters
        check_formulation_parameters(params, direct_solver, benders)
        
        # Analyze objective components
        analysis = analyze_objective_components(params, direct_sol, benders_sol)
        identify_key_differences(analysis)
        
        # Cross-validate solutions
        cross_results = cross_validate_solutions(params, direct_sol, benders_sol)
    
    return {
        'direct_sol': direct_sol,
        'benders_sol': benders_sol,
        'direct_obj': direct_obj,
        'benders_obj': benders_ub,
        'gap_pct': solution_comparison['objective_gap_pct']
    }


def main():
    """Run comprehensive diagnostics."""
    print("=== Production Planning Benders Decomposition Diagnostic ===")
    
    # Test with a small problem first
    print("\nCreating small test problem...")
    small_params = create_test_problem(
        num_periods=12,
        num_scenarios=2,
        capacity_to_demand_ratio=5,
        setup_to_holding_ratio=1000,
        seed=42
    )
    
    small_results = run_direct_vs_benders_test(small_params)
    
    # Test with a medium problem (like in your benchmark)
    print("\nCreating medium test problem...")
    medium_params = create_test_problem(
        num_periods=60,
        num_scenarios=3,
        capacity_to_demand_ratio=5,
        setup_to_holding_ratio=1000,
        seed=42
    )
    
    medium_results = run_direct_vs_benders_test(medium_params)
    
    print("\n=== Diagnostic testing complete ===")


if __name__ == "__main__":
    main()