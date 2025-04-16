import time
from typing import Dict, Set, Tuple, List
import gurobipy as gp
from gurobipy import GRB

from data.data_structures import ProblemParameters, TimingStats, Solution

class LaporteCutsGenerator:
    """
    Generator for Laporte-Louveaux cuts in the Benders decomposition algorithm.
    These cuts are particularly useful for stochastic integer programs with binary first-stage variables.
    """
    
    def __init__(self, params: ProblemParameters):
        """
        Initialize the Laporte cuts generator.
        
        Args:
            params: Problem parameters
        """
        self.params = params
        self.scenario_lower_bounds = {}  # Cache for scenario lower bounds
    
    def compute_scenario_lower_bound(self, scenario_idx: int) -> float:
        """
        Compute a lower bound on the objective value for a specific scenario.
        This is typically done by solving the scenario problem with all facilities available.
        
        Args:
            scenario_idx: Index of the scenario
            
        Returns:
            Lower bound on the objective value for the scenario
        """
        if scenario_idx in self.scenario_lower_bounds:
            return self.scenario_lower_bounds[scenario_idx]
        
        # Create a problem with all facilities located
        all_facilities = {i: True for i in range(self.params.total_periods)}
        
        # Create a temporary subproblem solver - avoiding circular imports here
        from solvers.scenario import MLScenarioSubproblem
        
        temp_solver = MLScenarioSubproblem(
            self.params.scenarios[scenario_idx],
            self.params,
            None  # No ML predictor needed for this
        )
        
        # Solve the subproblem with all facilities available
        obj_value, _ = temp_solver.solve_recursive(all_facilities)
        
        # Cache and return the bound
        self.scenario_lower_bounds[scenario_idx] = obj_value
        return obj_value
    
    def generate_laporte_cut(self, 
                            scenario_idx: int, 
                            current_solution: Dict[int, bool], 
                            subproblem_value: float) -> Tuple[List[float], float]:
        """
        Generate a Laporte-Louveaux cut for a specific scenario.
        
        The classic L-shaped method cut for binary first-stage variables has the form:
        z_s ≥ (v_s(x̂) - L_s) * (∑_{i∈X̂} x_i - ∑_{i∉X̂} x_i - |X̂| + 1) + L_s
        
        Where:
        - z_s is the value function variable for scenario s
        - v_s(x̂) is the optimal objective value for scenario s given solution x̂
        - L_s is a lower bound on the objective value for scenario s
        - X̂ is the set of facilities located in the current solution
        
        For our application, we use a modified form:
        z_s ≥ v_s(x̂) + ∑_{i∈V\X̂: b_i^s>0} (L_s - v_s(x̂)) * x_i
        
        This is stronger than the classic form and exploits the structure of our problem.
        
        Args:
            scenario_idx: Index of the scenario
            current_solution: Current solution of binary location variables
            subproblem_value: Optimal objective value for the scenario
            
        Returns:
            Tuple of (coefficients, constant term) for the cut
        """
        # Get the lower bound for this scenario
        L_s = self.compute_scenario_lower_bound(scenario_idx)
        
        # Identify located facilities
        located_facilities = set(i for i, is_located in current_solution.items() 
                               if is_located)
        
        # Initialize coefficients for each facility
        coefficients = [0.0] * self.params.total_periods
        
        # Set coefficients for non-located facilities with positive supply
        scenario = self.params.scenarios[scenario_idx]
        for i in range(self.params.total_periods):
            if i not in located_facilities:
                # Check if this facility has positive supply in this scenario
                if i < len(scenario.demands) and scenario.demands[i] > 0:
                    coefficients[i] = L_s - subproblem_value
        
        # The constant term is just the subproblem value
        constant_term = subproblem_value
        
        return coefficients, constant_term
    
    def add_laporte_cut(self, 
                       model: gp.Model, 
                       scenario_idx: int, 
                       fixed_setups: Dict[int, bool], 
                       objective_value: float, 
                       cut_counter: int) -> bool:
        """
        Add a Laporte-Louveaux cut to the master problem.
        
        Args:
            model: Gurobi model representing the master problem
            scenario_idx: Index of the scenario
            fixed_setups: Dictionary of fixed setup decisions
            objective_value: Optimal objective value for the scenario
            cut_counter: Counter for naming cuts
            
        Returns:
            True if a cut was added, False otherwise
        """
        if objective_value == float('inf'):
            # Cannot add an optimality cut for an infeasible subproblem
            return False
        
        # Generate the Laporte cut
        coefficients, constant_term = self.generate_laporte_cut(
            scenario_idx, fixed_setups, objective_value
        )
        
        # Get the value function variable for this scenario
        theta = model.getVarByName(f"theta[{scenario_idx}]")
        if theta is None:
            return False
        
        # Create the expression for the cut
        expr = gp.LinExpr(constant_term)
        for i, coef in enumerate(coefficients):
            if abs(coef) > 1e-10:  # Only add non-zero coefficients
                var = model.getVarByName(f"setup[{i}]")
                if var is not None:  # Only add terms for existing variables
                    expr.addTerms(coef, var)
        
        # Add the constraint: theta >= expr
        model.addConstr(
            theta >= expr,
            name=f"laporte_cut_{scenario_idx}_{cut_counter}"
        )
        
        return True