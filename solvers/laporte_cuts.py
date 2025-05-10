import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Tuple, List, Optional

from data.data_structures import ProblemParameters

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
    
    def generate_laporte_cut(
        self, 
        scenario_idx: int, 
        current_solution: Dict[int, bool], 
        subproblem_value: float
    ) -> Tuple[List[float], float]:
        """
        Generate a Laporte-Louveaux cut for a specific scenario.
        
        The classic L-shaped method cut for binary first-stage variables has the form:
        z_s ≥ (v_s(x̂) - L_s) * (∑_{i∈x̂} x_i - ∑_{i∉x̂} x_i - |x̂| + 1) + L_s
        
        Where:
        - z_s is the value function variable for scenario s
        - v_s(x̂) is the optimal objective value for scenario s given solution x̂
        - L_s is a lower bound on the objective value for scenario s
        - x̂ is the set of facilities located in the current solution
        
        For our application, we use a modified form:
        z_s ≥ v_s(x̂) + ∑_{i∈V\\x̂: b_i^s>0} (L_s - v_s(x̂)) * x_i
        
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
    
    def add_laporte_cut(self, model, scenario_idx, fixed_setups, objective_value, cut_counter):
        """Add a Laporte-Louveaux cut to the master problem.
        
        The Laporte-Louveaux cut for binary variables has the form:
        θ_s ≥ L_s + (v_s(xˆ) - L_s) * (Σ_{i∈xˆ} x_i - Σ_{i∉xˆ} x_i - |xˆ| + 1)
        
        This ensures that:
        - When x = xˆ, the cut enforces θ_s ≥ v_s(xˆ)
        - For all other binary x, the cut enforces θ_s ≥ L_s
        """
        if objective_value == float('inf'):
            return None
        
        # Get the value function variable for this scenario
        theta = model.getVarByName(f"theta[{scenario_idx}]")
        if theta is None:
            return None
        
        # Get the lower bound for this scenario
        L_s = self.compute_scenario_lower_bound(scenario_idx)
        
        # Collect the periods with setup = 1 in current solution
        setup_periods = set(t for t, is_setup in fixed_setups.items() if is_setup)
        
        # Calculate (v_s(xˆ) - L_s) coefficient
        value_diff = objective_value - L_s
        
        # Create the combinatorial expression
        expr = gp.LinExpr()
        
        # Add terms for ALL linking periods
        for t in self.params.linking_periods:
            var = model.getVarByName(f"setup[{t}]")
            if var is not None:
                if t in setup_periods:
                    # For x_t = 1 in current solution
                    expr.addTerms(1.0, var)
                else:
                    # For x_t = 0 in current solution
                    expr.addTerms(-1.0, var)
        
        # Subtract |xˆ| and add 1
        expr.addConstant(-len(setup_periods) + 1.0)
        
        # Create the full cut: θ_s ≥ L_s + (v_s(xˆ) - L_s) * expr
        cut = L_s + value_diff * expr
        
        # Add the constraint
        laporte_constr = model.addConstr(
            theta >= cut,
            name=f"laporte_cut_{scenario_idx}_{cut_counter}"
        )
        
        # Set the Lazy attribute
        laporte_constr.Lazy = 1
        
        return laporte_constr