import time
from typing import Tuple, Dict, Set

import gurobipy as gp
from gurobipy import GRB

from data.data_structures import ProblemParameters, TimingStats, Solution
from models.ml_predictor import MLSubproblemPredictor
from solvers.scenario import MLScenarioSubproblem

class MLBendersDecomposition:
    """Benders decomposition solver with ML enhancement."""
    
    def __init__(self, params: ProblemParameters, use_ml: bool = True):
        """Initialize the Benders decomposition solver.
        
        Args:
            params: Problem parameters
            use_ml: Whether to use ML prediction
        """
        self.params = params
        self.use_ml = use_ml
        self.ml_predictor = MLSubproblemPredictor() if use_ml else None
        
        # Create scenario problems
        self.scenario_problems = [
            MLScenarioSubproblem(s, params, self.ml_predictor) 
            for s in params.scenarios
        ]
        
        self.master_model = None
        self.timing_stats = None
        self.best_solution = None
        self.num_feasibility_cuts = 0
    
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
            min_setups = max(1, total_demand // self.params.capacity + 
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
    
    def add_combinatorial_cut(self, scenario_idx: int, critical_setups: Set[int],
                            current_setups: Dict[int, bool]):
        """Add logic-based Benders optimality cut.
        
        Args:
            scenario_idx: Index of the scenario
            critical_setups: Set of critical setup periods
            current_setups: Dictionary of current setup decisions
        """
        if not critical_setups:
            return
            
        expr = gp.LinExpr()
        zero_setups = set()
        
        for t in self.params.linking_periods:
            if not current_setups[t]:
                var = self.master_model.getVarByName(f"setup[{t}]")
                expr += var
                zero_setups.add(t)
        
        if zero_setups:
            theta = self.master_model.getVarByName(f"theta[{scenario_idx}]")
            min_cost = len(critical_setups) * self.params.fixed_cost
            
            self.master_model.addConstr(
                theta >= min_cost * (1 - expr/len(zero_setups))
            )
    
    def add_feasibility_cut(self, critical_setups: Set[int], current_setups: Dict[int, bool]):
        """Add a pure feasibility combinatorial Benders cut.
        
        Args:
            critical_setups: Set of critical setup periods
            current_setups: Dictionary of current setup decisions
            
        Returns:
            True if a cut was added, False otherwise
        """
        if not critical_setups:
            return False
        
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
    
    def add_no_good_cut(self, current_setups: Dict[int, bool]):
        """Add a no-good cut when we can't determine specific critical setups.
        
        Args:
            current_setups: Dictionary of current setup decisions
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
    
    def solve(self, max_iterations: int = 100, 
             tolerance: float = 1e-6) -> Tuple[float, float, TimingStats]:
        """Solve the production planning problem using Benders decomposition.
        
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
        
        self.build_master_problem()
        best_lb = float('-inf')
        best_ub = float('inf')
        best_solution_data = None
        iteration = 0
        self.num_feasibility_cuts = 0

        while iteration < max_iterations and (best_ub - best_lb) > tolerance:
            # Solve master problem
            master_start = time.time()
            self.master_model.optimize()
            master_time += time.time() - master_start
            
            if self.master_model.status != GRB.OPTIMAL:
                raise ValueError("Master problem infeasible")

            current_lb = self.master_model.objVal
            
            # Get setup decisions for linking periods
            fixed_setups = {}
            for t in self.params.linking_periods:
                var = self.master_model.getVarByName(f"setup[{t}]")
                fixed_setups[t] = var.x > 0.5

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
                
                cost, solution_data = scenario_prob.solve_recursive(fixed_setups)
                
                if cost == float('inf'):
                    all_feasible = False
                    critical_setups = scenario_prob.get_critical_setups()
                    
                    # Add both types of cuts
                    added_feas = self.add_feasibility_cut(critical_setups, fixed_setups)
                    added_opt = self.add_combinatorial_cut(s_idx, critical_setups, fixed_setups)
                    
                    # If we couldn't add a cut, we need to prevent this solution another way
                    if not (added_feas or added_opt):
                        # Add a generic no-good cut
                        self.add_no_good_cut(fixed_setups)
                    
                    break
                
                total_cost += scenario.probability * cost
                current_solution['setup'].append(solution_data['setup'])
                current_solution['production'].append(solution_data['production'])
                current_solution['inventory'].append(solution_data['inventory'])
            
            subproblem_time += time.time() - subprob_start
            
            if self.use_ml:
                ml_prediction_time += time.time() - ml_start

            if all_feasible:
                if total_cost < best_ub:
                    best_ub = total_cost
                    best_solution_data = current_solution

            if current_lb > best_lb:
                best_lb = current_lb

            iteration += 1

        total_solve_time = time.time() - solve_start
        
        self.timing_stats = TimingStats(
            total_solve_time=total_solve_time,
            subproblem_time=subproblem_time,
            master_time=master_time,
            num_iterations=iteration,
            ml_prediction_time=ml_prediction_time
        )
        
        if best_solution_data is None:
            raise ValueError("No feasible solution found")
            
        self.best_solution = Solution(
            setup=best_solution_data['setup'],
            production=best_solution_data['production'],
            inventory=best_solution_data['inventory'],
            objective=best_ub
        )
        
        return best_lb, best_ub, self.timing_stats