from typing import Dict, Tuple, Set
import gurobipy as gp
from gurobipy import GRB
import time
import torch
import numpy as np

from data.data_structures import ScenarioData, ProblemParameters
from models.ml_predictor import MLSubproblemPredictor

class MLTimeBlockSubproblem:
    """Time block subproblem solver with ML enhancement."""
    
    def __init__(self, start_period: int, num_periods: int, 
                 scenario_data: ScenarioData, params: ProblemParameters,
                 ml_predictor: MLSubproblemPredictor = None):
        """Initialize the time block subproblem.
        
        Args:
            start_period: Starting period of the time block
            num_periods: Number of periods in the time block
            scenario_data: Scenario data for the subproblem
            params: Problem parameters
            ml_predictor: ML predictor for solution prediction
        """
        self.start_period = start_period
        self.num_periods = num_periods
        self.scenario_data = scenario_data
        self.params = params
        self.ml_predictor = ml_predictor
        self.model = None
        self.setup_vars = {}
        self.prod_vars = {}
        self.inv_vars = {}
        self.ml_solution = None
    
    def predict_solution(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """Use ML to predict the solution instead of solving with Gurobi.
        
        Args:
            initial_inventory: Initial inventory level
            fixed_setups: Dictionary of setup decisions fixed by the master problem
            
        Returns:
            Tuple of (cost, is_feasible)
        """
        if self.ml_predictor is None:
            # No ML predictor available, skip prediction
            return float('inf'), False
        
        if not self.ml_predictor.is_loaded or self.ml_predictor.model is None:
            # ML model not loaded, skip prediction
            return float('inf'), False
            
        try:
            # Prepare features
            features = self.ml_predictor.prepare_features(
                fixed_setups=fixed_setups,
                scenario=self.scenario_data  # Pass the entire scenario object
            )
            
            # Get prediction
            setup_pred, production_plan, inventory_plan = self.ml_predictor.predict_subproblem_solution(
                features, self.num_periods)
            
            # Check if prediction succeeded
            if setup_pred is None or production_plan is None or inventory_plan is None:
                print("ML prediction returned None values. Falling back to Gurobi.")
                return float('inf'), False
            
            # Store solution
            self.ml_solution = {
                'setup': [fixed_setups.get(self.start_period + t, False) for t in range(self.num_periods)],
                'production': production_plan.tolist() if isinstance(production_plan, np.ndarray) else production_plan,
                'inventory': (list(inventory_plan) if isinstance(inventory_plan, np.ndarray) else inventory_plan) + [0.0]  # Add final inventory
            }
            
            # Calculate approximate cost
            cost = 0.0
            for t in range(self.num_periods):
                period = self.start_period + t
                if period in fixed_setups and fixed_setups[period]:
                    cost += self.params.fixed_cost
                cost += production_plan[t] * self.params.production_cost
                cost += inventory_plan[t] * self.params.holding_cost
            
            # Verify solution feasibility
            is_feasible = self.verify_solution_feasibility(
                initial_inventory, fixed_setups, 
                self.ml_solution['setup'], production_plan, inventory_plan
            )
            
            if is_feasible:
                return cost, True
            else:
                print("ML predicted solution is not feasible. Falling back to Gurobi.")
                self.ml_solution = None
                return float('inf'), False
                
        except Exception as e:
            print(f"ML prediction error: {str(e)}")
            self.ml_solution = None
            return float('inf'), False
    
    def verify_solution_feasibility(self, initial_inv, fixed_setups, setup, production, inventory):
        """Verify if the ML-predicted solution is feasible.
        
        Args:
            initial_inv: Initial inventory level
            fixed_setups: Dictionary of setup decisions fixed by the master problem
            setup: Predicted setup decisions
            production: Predicted production quantities
            inventory: Predicted inventory levels
            
        Returns:
            True if the solution is feasible, False otherwise
        """
        # Check capacity constraints
        for t in range(self.num_periods):
            if production[t] > self.params.capacity * (1 + 1e-6):  # Small tolerance
                return False
            
            # If setup is required by master problem, make sure we respect it
            period = self.start_period + t
            if period in self.params.linking_periods:
                master_setup = fixed_setups[period]
                if master_setup and not setup[t]:
                    return False
                elif not master_setup and setup[t]:
                    return False
            
            # Production with no setup
            if production[t] > 1e-6 and not setup[t]:
                return False
        
        # Check flow balance constraints
        inv = initial_inv
        for t in range(self.num_periods):
            period = self.start_period + t
            demand = self.scenario_data.demands[period]
            
            inv = inv + production[t] - demand
            
            # Allow small numerical errors
            if abs(inv - inventory[t]) > 1e-4:
                return False
            
            # Inventory can't be negative
            if inv < -1e-6:
                return False
        
        return True
    
    def build_model(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """Build Gurobi model as fallback if ML prediction fails.
        
        Args:
            initial_inventory: Initial inventory level
            fixed_setups: Dictionary of setup decisions fixed by the master problem
            
        Returns:
            Gurobi model
        """
        model = gp.Model("TimeBlock")
        model.setParam('OutputFlag', 0)
        
        # Variables
        for t in range(self.num_periods):
            period = self.start_period + t
            if period in self.params.linking_periods:
                # Setup fixed by master problem
                self.setup_vars[t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{t}]")
                self.setup_vars[t].lb = self.setup_vars[t].ub = int(fixed_setups[period])
            else:
                # Setup decided here
                self.setup_vars[t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{t}]")
            
            self.prod_vars[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"production[{t}]")
        
        for t in range(self.num_periods + 1):
            self.inv_vars[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"inventory[{t}]")
        
        # Fix initial inventory
        self.inv_vars[0].lb = initial_inventory
        self.inv_vars[0].ub = initial_inventory
        
        # Constraints
        for t in range(self.num_periods):
            period = self.start_period + t
            
            # Inventory balance
            model.addConstr(
                self.inv_vars[t] + self.prod_vars[t] == 
                self.scenario_data.demands[period] + self.inv_vars[t+1],
                name=f"balance_{t}"
            )
            
            # Capacity constraint
            model.addConstr(
                self.prod_vars[t] <= self.params.capacity * self.setup_vars[t],
                name=f"capacity_{t}"
            )
        
        # Objective
        obj = (gp.quicksum(self.params.fixed_cost * self.setup_vars[t] + 
                          self.params.production_cost * self.prod_vars[t] + 
                          self.params.holding_cost * self.inv_vars[t]
                          for t in range(self.num_periods)))
        model.setObjective(obj, GRB.MINIMIZE)
        
        self.model = model
        return model
    
    def solve(self) -> Tuple[float, bool]:
        """Solve using Gurobi (fallback).
        
        Returns:
            Tuple of (objective_value, is_feasible)
        """
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            return self.model.objVal, True
        return float('inf'), False
    
    def get_solution(self) -> Dict:
        """Get solution (either from ML or Gurobi).
        
        Returns:
            Dictionary containing the solution
        """
        if self.ml_solution is not None:
            return self.ml_solution
        
        # Fall back to Gurobi solution
        return {
            'setup': [self.setup_vars[t].x > 0.5 for t in range(self.num_periods)],
            'production': [self.prod_vars[t].x for t in range(self.num_periods)],
            'inventory': [self.inv_vars[t].x for t in range(self.num_periods + 1)]
        }
    
    def get_critical_setups(self) -> Set[int]:
        """Identify critical setup periods.
        
        Returns:
            Set of critical setup periods
        """
        critical_periods = set()
        
        # Get solution (either ML or Gurobi)
        sol = self.get_solution()
        
        for t in range(self.num_periods):
            period = self.start_period + t
            if period not in self.params.linking_periods:
                if sol['setup'][t]:
                    # Check if setup is critical (significant production)
                    if sol['production'][t] > 0.1 * self.params.capacity:
                        critical_periods.add(period)
        return critical_periods