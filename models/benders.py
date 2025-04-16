import time
from typing import Dict, Tuple, Set, List
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from data.data_structures import ProblemParameters, TimingStats, Solution
from models.ml_predictor import MLSubproblemPredictor
from solvers.scenario import MLScenarioSubproblem
from solvers.laporte_cuts import LaporteCutsGenerator

class MLBendersDecomposition:
    """Improved Benders decomposition with trust region, enhanced cuts, and Laporte cuts."""
    
    def __init__(self, params: ProblemParameters, use_ml: bool = True, 
                use_trust_region: bool = True, use_laporte_cuts: bool = True):
        """Initialize the improved Benders decomposition solver.
        
        Args:
            params: Problem parameters
            use_ml: Whether to use ML prediction
            use_trust_region: Whether to use trust region constraints
            use_laporte_cuts: Whether to use Laporte cuts
        """
        self.params = params
        self.use_ml = use_ml
        self.use_trust_region = use_trust_region
        self.use_laporte_cuts = use_laporte_cuts
        self.ml_predictor = MLSubproblemPredictor() if use_ml else None
        
        # Create scenario problems
        self.scenario_problems = [
            MLScenarioSubproblem(s, params, self.ml_predictor) 
            for s in params.scenarios
        ]
        
        # Initialize Laporte cuts generator if enabled
        self.laporte_cuts_generator = LaporteCutsGenerator(params) if use_laporte_cuts else None
        
        self.master_model = None
        self.timing_stats = None
        self.best_solution = None
        self.previous_master_solutions = []  # Store previous master solutions for trust region
        
        # Statistics
        self.num_feasibility_cuts = 0
        self.num_optimality_cuts = 0
        self.num_trust_region_cuts = 0
        self.num_laporte_cuts = 0  # Track Laporte cuts
    
    def build_master_problem(self):
        """Build the master problem for Benders decomposition.
        
        Returns:
            Gurobi model
        """
        model = gp.Model("Master")
        model.setParam('OutputFlag', 0)

        # Setup variables only for linking periods
        setup = {}
        for t in self.params.linking_periods:
            setup[t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{t}]")

        # Theta variables for expected cost
        theta = {}
        for s in range(len(self.params.scenarios)):
            theta[s] = model.addVar(lb=0, name=f"theta[{s}]")

        # Initial cuts for minimum cost
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            total_demand = sum(scenario.demands)
            min_setups = max(1, int(total_demand // self.params.capacity) + 
                           (1 if total_demand % self.params.capacity > 0 else 0))
            
            model.addConstr(
                theta[s] >= (min_setups * self.params.fixed_cost + 
                            total_demand * self.params.production_cost)
            )

        # Objective
        obj = gp.quicksum(self.params.scenarios[s].probability * theta[s] 
                         for s in range(len(self.params.scenarios)))
        model.setObjective(obj, GRB.MINIMIZE)

        self.master_model = model
        return model
    
    def add_trust_region_constraint(self, iteration):
        """Add a simple trust region constraint.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            True if added successfully, False otherwise
        """
        if not self.use_trust_region or iteration == 0 or not self.previous_master_solutions:
            return False
            
        # Remove any existing trust region constraints to avoid accumulation
        for c in self.master_model.getConstrs():
            if "trust_region" in c.ConstrName:
                self.master_model.remove(c)
                
        # Get the most recent master solution
        prev_setups = self.previous_master_solutions[-1]
        
        # Create Hamming distance expression
        diff_expr = gp.LinExpr()
        
        for t in self.params.linking_periods:
            var = self.master_model.getVarByName(f"setup[{t}]")
            if prev_setups[t]:
                # Was 1, count if it becomes 0
                diff_expr += (1 - var)
            else:
                # Was 0, count if it becomes 1
                diff_expr += var
        
        # Calculate allowed changes - start small and gradually increase
        base_radius = max(1, len(self.params.linking_periods) // 10)  # At least 1, or 10% of variables
        radius_growth = min(iteration // 3, 3)  # +1 every 3 iterations, max +3
        max_changes = base_radius + radius_growth
        
        # Don't over-restrict - cap at 1/3 of variables
        max_changes = min(max_changes, len(self.params.linking_periods) // 3)
        
        # Add constraint
        self.master_model.addConstr(
            diff_expr <= max_changes,
            name=f"trust_region_{iteration}"
        )
        self.num_trust_region_cuts += 1
        return True
    
    def extract_master_solution(self):
        """Extract setup decisions from master problem.
        
        Returns:
            Dictionary of setup decisions
        """
        fixed_setups = {}
        
        for t in self.params.linking_periods:
            var = self.master_model.getVarByName(f"setup[{t}]")
            fixed_setups[t] = var.x > 0.5
        
        # Store solution for trust region in next iteration
        self.previous_master_solutions.append(fixed_setups)
        
        # Keep only the last few solutions to save memory
        if len(self.previous_master_solutions) > 5:
            self.previous_master_solutions.pop(0)
        
        return fixed_setups
    
    def add_feasibility_cut(self, critical_setups: Set[int], current_setups: Dict[int, bool]):
        """Add a pure feasibility combinatorial Benders cut.
        
        Args:
            critical_setups: Set of critical setup periods
            current_setups: Dictionary of current setup decisions
            
        Returns:
            True if a cut was added, False otherwise
        """
        if not critical_setups:
            return self.add_no_good_cut(current_setups)
        
        # Create expression that counts how many setup decisions match the current infeasible solution
        match_expr = gp.LinExpr()
        decision_count = 0
        
        for t in self.params.linking_periods:
            var = self.master_model.getVarByName(f"setup[{t}]")
            if current_setups[t]:
                # Setup is 1 in current infeasible solution
                match_expr += var
                decision_count += 1
            else:
                # Setup is 0 in current infeasible solution
                match_expr += (1 - var)
                decision_count += 1
        
        # Add constraint: at least one decision must be different
        self.master_model.addConstr(
            match_expr <= decision_count - 1,
            name=f"feasibility_cut_{self.num_feasibility_cuts}"
        )
        self.num_feasibility_cuts += 1
        return True
    
    def add_improved_optimality_cut(self, scenario_idx: int, critical_setups: Set[int],
                                  current_setups: Dict[int, bool], solution_cost: float):
        """Add an improved optimality cut with better sensitivity information.
        
        Args:
            scenario_idx: Index of the scenario
            critical_setups: Set of critical setup periods
            current_setups: Dictionary of current setup decisions
            solution_cost: Cost of the current solution
            
        Returns:
            True if a cut was added, False otherwise
        """
        theta = self.master_model.getVarByName(f"theta[{scenario_idx}]")
        
        # Calculate effective contribution of each setup
        setup_cost_contrib = {}
        
        # For each linking period, estimate its cost contribution
        for t in self.params.linking_periods:
            if t in critical_setups:
                # Critical setup - turning it off would increase cost
                setup_cost_contrib[t] = self.params.fixed_cost * 0.99  # Conservative estimate
            else:
                # Non-critical setup - turning it on would add cost
                setup_cost_contrib[t] = -self.params.fixed_cost * 0.99  # Conservative estimate
        
        # Build cut expression
        cut_expr = solution_cost  # Start with current solution cost
        
        for t in self.params.linking_periods:
            var = self.master_model.getVarByName(f"setup[{t}]")
            # Add term that adjusts cost based on changing this setup decision
            current_val = 1.0 if current_setups[t] else 0.0
            cut_expr += setup_cost_contrib[t] * (var - current_val)
        
        # Add cut: theta >= cut_expr
        self.master_model.addConstr(
            theta >= cut_expr,
            name=f"improved_optimality_cut_{self.num_optimality_cuts}"
        )
        self.num_optimality_cuts += 1
        return True
    
    def add_traditional_optimality_cut(self, scenario_idx: int, critical_setups: Set[int],
                                     current_setups: Dict[int, bool]):
        """Add a traditional logic-based Benders optimality cut.
        
        Args:
            scenario_idx: Index of the scenario
            critical_setups: Set of critical setup periods
            current_setups: Dictionary of current setup decisions
            
        Returns:
            True if a cut was added, False otherwise
        """
        if not critical_setups:
            return False
            
        # Find which critical setups are OFF in current solution
        off_setups = set()
        for t in critical_setups:
            if t in current_setups and not current_setups[t]:
                off_setups.add(t)
        
        if not off_setups:
            return False  # No critical setups are off
        
        expr = gp.LinExpr()
        
        for t in off_setups:
            var = self.master_model.getVarByName(f"setup[{t}]")
            expr += var
        
        theta = self.master_model.getVarByName(f"theta[{scenario_idx}]")
        min_cost = len(off_setups) * self.params.fixed_cost
        
        self.master_model.addConstr(
            theta >= min_cost * (1 - expr/len(off_setups)),
            name=f"traditional_optimality_cut_{self.num_optimality_cuts}"
        )
        self.num_optimality_cuts += 1
        return True
    
    def add_no_good_cut(self, current_setups: Dict[int, bool]):
        """Add a no-good cut when we can't determine specific critical setups.
        
        Args:
            current_setups: Dictionary of current setup decisions
            
        Returns:
            True if a cut was added, False otherwise
        """
        match_expr = gp.LinExpr()
        decision_count = 0
        
        for t in self.params.linking_periods:
            var = self.master_model.getVarByName(f"setup[{t}]")
            if current_setups[t]:
                match_expr += var
                decision_count += 1
            else:
                match_expr += (1 - var)
                decision_count += 1
        
        self.master_model.addConstr(
            match_expr <= decision_count - 1,
            name=f"no_good_cut_{self.num_feasibility_cuts}"
        )
        self.num_feasibility_cuts += 1
        return True
    
    def solve(self, max_iterations: int = 100, 
             tolerance: float = 1e-6) -> Tuple[float, float, TimingStats]:
        """Solve the production planning problem using improved Benders decomposition.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Optimality tolerance
            
        Returns:
            Tuple of (lower_bound, upper_bound, timing_stats)
            
        Raises:
            ValueError: If no feasible solution is found or the master problem is infeasible
        """
        solve_start = time.time()
        subproblem_time = 0
        master_time = 0
        ml_prediction_time = 0
        laporte_cuts_time = 0  # Track time spent generating Laporte cuts
        
        self.build_master_problem()
        best_lb = float('-inf')
        best_ub = float('inf')
        best_solution_data = None
        iteration = 0
        self.num_feasibility_cuts = 0
        self.num_optimality_cuts = 0
        self.num_trust_region_cuts = 0
        self.num_laporte_cuts = 0
        
        # Track iterations without improvement
        stagnant_iterations = 0
        last_lb = float('-inf')

        while iteration < max_iterations and (best_ub - best_lb) > tolerance:
            try:
                # Add trust region constraint if enabled
                # if iteration > 0 and self.use_trust_region:
                #     self.add_trust_region_constraint(iteration)
                
                # Solve master zproblem
                master_start = time.time()
                self.master_model.optimize()
                master_time += time.time() - master_start
                
                # Check master problem status
                if self.master_model.status != GRB.OPTIMAL:
                    if iteration == 0:
                        raise ValueError(f"Master problem infeasible")
                    else:
                        print(f"Master problem not optimal at iteration {iteration}. Status: {self.master_model.status}")
                        print("Using best solution found.")
                        break
                
                # Update lower bound
                current_lb = self.master_model.objVal
                
                # Check for lower bound improvement
                if current_lb > last_lb + 1e-6:  # Small tolerance for numerical issues
                    stagnant_iterations = 0
                    last_lb = current_lb
                else:
                    stagnant_iterations += 1
                
                # Get master solution
                fixed_setups = self.extract_master_solution()
                
                # Solve scenario subproblems
                ml_start = time.time()
                subprob_start = time.time()
                total_cost = 0
                current_solution = {
                    'setup': [],
                    'production': [],
                    'inventory': []
                }
                all_feasible = True
                
                for s_idx, (scenario_prob, scenario) in enumerate(zip(
                    self.scenario_problems, self.params.scenarios)):
                    
                    # Get current theta value
                    theta_var = self.master_model.getVarByName(f"theta[{s_idx}]")
                    theta_val = theta_var.x
                    
                    # Solve the subproblem
                    cost, solution_data = scenario_prob.solve_recursive(fixed_setups)
                    
                    if cost == float('inf'):
                        all_feasible = False
                        critical_setups = scenario_prob.get_critical_setups()
                        
                        # Add feasibility cut
                        self.add_feasibility_cut(critical_setups, fixed_setups)
                        break
                    
                    # Check if we need to add optimality cuts
                    if theta_val < cost - tolerance:
                        critical_setups = scenario_prob.get_critical_setups()
                        
                        # Add both traditional and improved optimality cuts for better convergence
                        self.add_traditional_optimality_cut(s_idx, critical_setups, fixed_setups)
                        self.add_improved_optimality_cut(s_idx, critical_setups, fixed_setups, cost)
                        
                        # Add Laporte cuts if enabled
                        if self.use_laporte_cuts and self.laporte_cuts_generator:
                            laporte_start = time.time()
                            if self.laporte_cuts_generator.add_laporte_cut(
                                self.master_model, s_idx, fixed_setups, cost, self.num_laporte_cuts):
                                self.num_laporte_cuts += 1
                            laporte_cuts_time += time.time() - laporte_start
                    
                    # Update total cost
                    total_cost += scenario.probability * cost
                    
                    # Store solution
                    current_solution['setup'].append(solution_data['setup'])
                    current_solution['production'].append(solution_data['production'])
                    current_solution['inventory'].append(solution_data['inventory'])
                
                subproblem_time += time.time() - subprob_start
                
                if self.use_ml:
                    ml_prediction_time += time.time() - ml_start
                
                # Update bounds if feasible solution found
                if all_feasible:
                    if total_cost < best_ub:
                        best_ub = total_cost
                        best_solution_data = current_solution
                
                if current_lb > best_lb:
                    best_lb = current_lb
                
                # Print progress every 5 iterations
                iteration += 1
                if iteration % 5 == 0 or iteration == 1:
                    gap = ((best_ub - best_lb) / best_ub * 100) if best_ub < float('inf') else float('inf') 
                    print(f"Iteration {iteration}: LB={best_lb:.2f}, UB={best_ub:.2f}, Gap={gap:.2f}%")
                
                # Check if we're stagnating for too long
                if stagnant_iterations > 10:
                    print(f"No progress in lower bound for {stagnant_iterations} iterations. Stopping.")
                    break
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                iteration += 1
                continue
        
        total_solve_time = time.time() - solve_start
        
        # Check if we found any feasible solution
        if best_solution_data is None:
            raise ValueError("No feasible solution found")
        
        # Create an updated TimingStats object that includes laporte_cuts_time
        self.timing_stats = TimingStats(
            total_solve_time=total_solve_time,
            subproblem_time=subproblem_time,
            master_time=master_time,
            num_iterations=iteration,
            ml_prediction_time=ml_prediction_time,
            laporte_cuts_time=laporte_cuts_time  # Add Laporte cuts time
        )
        
        self.best_solution = Solution(
            setup=best_solution_data['setup'],
            production=best_solution_data['production'],
            inventory=best_solution_data['inventory'],
            objective=best_ub
        )
        
        # Print statistics
        print(f"Benders iterations: {iteration}")
        print(f"Feasibility cuts added: {self.num_feasibility_cuts}")
        print(f"Optimality cuts added: {self.num_optimality_cuts}")
        if self.use_trust_region:
            print(f"Trust region cuts added: {self.num_trust_region_cuts}")
        if self.use_laporte_cuts:
            print(f"Laporte cuts added: {self.num_laporte_cuts}")
        
        return best_lb, best_ub, self.timing_stats