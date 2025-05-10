import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

from data.data_structures import ProblemParameters, TimingStats, Solution

class DirectSolver:
    """Direct solver that solves the entire problem in one go without decomposition."""
    
    def __init__(self, params: ProblemParameters):
        """Initialize the direct solver.
        
        Args:
            params: Problem parameters
        """
        self.params = params
        self.model = None
        self.timing_stats = None
        self.best_solution = None
        
        # Added parameter to force setup decisions at all linking periods
        self.fixed_linking_values = {}
        # Force ALL linking periods to TRUE to match Benders behavior
        for t in self.params.linking_periods:
            self.fixed_linking_values[t] = True
    
    def build_model(self):
        """Build the complete optimization model with fixed linking period values."""
        model = gp.Model("Direct")
        model.setParam('OutputFlag', 0)
        
        # Set tighter tolerances for better convergence
        model.setParam('MIPGap', 1e-9)
        model.setParam('IntFeasTol', 1e-9)
        
        # Create variables for all scenarios and periods
        setup = {}
        production = {}
        inventory = {}
        
        # Create here-and-now variables first
        here_setup = {}
        here_production = {}
        
        for t in sorted(self.params.linking_periods):
            # Check if this linking period should be fixed
            if t in self.fixed_linking_values:
                is_fixed = True
                fixed_value = self.fixed_linking_values[t]
            else:
                is_fixed = False
                fixed_value = None
                
            # Create the variable
            here_setup[t] = model.addVar(vtype=GRB.BINARY, name=f"here_setup_{t}")
            
            # If fixed, set bounds
            if is_fixed:
                here_setup[t].lb = 1.0 if fixed_value else 0.0
                here_setup[t].ub = 1.0 if fixed_value else 0.0
                
            here_production[t] = model.addVar(lb=0, name=f"here_production_{t}")
            
            # Add capacity constraint
            model.addConstr(
                here_production[t] <= self.params.capacity * here_setup[t],
                name=f"here_capacity_{t}"
            )
        
        # Update model to register the here-and-now variables
        model.update()
        
        # Scenario-specific variables
        for s in range(len(self.params.scenarios)):
            setup[s] = {}
            production[s] = {}
            inventory[s] = {}
            
            for t in range(self.params.total_periods):
                setup[s][t] = model.addVar(vtype=GRB.BINARY, name=f"setup_{s}_{t}")
                production[s][t] = model.addVar(lb=0, name=f"production_{s}_{t}")
                inventory[s][t] = model.addVar(lb=0, name=f"inventory_{s}_{t}")
                
                # Add capacity constraint
                model.addConstr(
                    production[s][t] <= self.params.capacity * setup[s][t],
                    name=f"capacity_{s}_{t}"
                )
                
                # For linking periods, enforce non-anticipativity
                if t in self.params.linking_periods:
                    model.addConstr(
                        setup[s][t] == here_setup[t],
                        name=f"non_ant_setup_{s}_{t}"
                    )
                    model.addConstr(
                        production[s][t] == here_production[t],
                        name=f"non_ant_prod_{s}_{t}"
                    )
        
        # Flow balance constraints
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            
            # Initial inventory (assume zero initial inventory if not specified)
            initial_inventory = getattr(self.params, 'initial_inventory', 0.0)
            
            # First period
            model.addConstr(
                inventory[s][0] == initial_inventory + production[s][0] - scenario.demands[0],
                name=f"flow_balance_initial_{s}"
            )
            
            # Remaining periods
            for t in range(1, self.params.total_periods):
                model.addConstr(
                    inventory[s][t] == inventory[s][t-1] + production[s][t] - scenario.demands[t],
                    name=f"flow_balance_{s}_{t}"
                )
        
        # Add valid inequalities to strengthen the formulation
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            
            # Minimum setups needed based on total demand
            total_demand = sum(scenario.demands)
            min_setups = max(1, int(total_demand / self.params.capacity) + 
                          (1 if total_demand % self.params.capacity > 0 else 0))
            
            setup_sum = gp.quicksum(setup[s][t] for t in range(self.params.total_periods))
            model.addConstr(setup_sum >= min_setups, name=f"min_setups_{s}")
            
            # Forward-looking demand coverage
            for t in range(self.params.total_periods):
                # Calculate remaining demand from this period forward
                remaining_demand = sum(scenario.demands[t:])
                if remaining_demand > 0:
                    # Minimum setups needed for remaining demand
                    remaining_min_setups = max(1, int(remaining_demand / self.params.capacity) + 
                                          (1 if remaining_demand % self.params.capacity > 0 else 0))
                    
                    remaining_setup_sum = gp.quicksum(setup[s][i] for i in range(t, self.params.total_periods))
                    model.addConstr(remaining_setup_sum >= remaining_min_setups, 
                                  name=f"remaining_setups_{s}_{t}")
        
        # Objective function: minimize expected total cost
        obj = gp.LinExpr()
        
        # Debug prints
        print("Debugging objective function calculations:")
        print(f"Fixed cost: {self.params.fixed_cost}")
        print(f"Production cost: {self.params.production_cost}")
        print(f"Holding cost: {self.params.holding_cost}")
        print(f"Fixed linking periods: {self.fixed_linking_values}")
        
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            print(f"Scenario {s} probability: {scenario.probability}")
            
            # Setup costs
            setup_cost = gp.quicksum(self.params.fixed_cost * setup[s][t] 
                                  for t in range(self.params.total_periods))
            
            # Production costs
            prod_cost = gp.quicksum(self.params.production_cost * production[s][t] 
                                  for t in range(self.params.total_periods))
            
            # Inventory holding costs
            hold_cost = gp.quicksum(self.params.holding_cost * inventory[s][t] 
                                 for t in range(self.params.total_periods))
            
            # Total cost for this scenario
            scenario_cost = setup_cost + prod_cost + hold_cost
            
            # Add to expected cost with appropriate probability
            obj += scenario.probability * scenario_cost
        
        model.setObjective(obj, GRB.MINIMIZE)
        self.model = model
        return model
    
    def solve(self) -> Tuple[float, TimingStats]:
        """Solve the production planning problem using the direct approach.
        
        Returns:
            Tuple of (objective_value, timing_stats)
            
        Raises:
            ValueError: If no feasible solution is found
        """
        solve_start = time.time()
        
        # Build and solve model
        self.build_model()
        self.model.optimize()
        
        # Check status
        if self.model.status != GRB.OPTIMAL:
            raise ValueError(f"Failed to find optimal solution. Status: {self.model.status}")
        
        # Extract solution
        setup_solution = []
        production_solution = []
        inventory_solution = []
        
        # Calculate scenario costs for debugging
        scenario_costs = []
        
        for s in range(len(self.params.scenarios)):
            setup_s = []
            production_s = []
            inventory_s = []
            
            # Track costs for validation
            setup_cost = 0
            prod_cost = 0
            hold_cost = 0
            
            for t in range(self.params.total_periods):
                setup_var = self.model.getVarByName(f"setup_{s}_{t}")
                production_var = self.model.getVarByName(f"production_{s}_{t}")
                inventory_var = self.model.getVarByName(f"inventory_{s}_{t}")
                
                is_setup = setup_var.x > 0.5
                setup_s.append(is_setup)
                production_s.append(production_var.x)
                inventory_s.append(inventory_var.x)
                
                # Calculate costs
                if is_setup:
                    setup_cost += self.params.fixed_cost
                prod_cost += self.params.production_cost * production_var.x
                hold_cost += self.params.holding_cost * inventory_var.x
            
            setup_solution.append(setup_s)
            production_solution.append(production_s)
            inventory_solution.append(inventory_s)
            
            # Track total scenario cost
            total_cost = setup_cost + prod_cost + hold_cost
            weighted_cost = total_cost * self.params.scenarios[s].probability
            scenario_costs.append({
                'scenario': s,
                'setup': setup_cost,
                'production': prod_cost,
                'holding': hold_cost,
                'total': total_cost,
                'weight': self.params.scenarios[s].probability,
                'weighted': weighted_cost
            })
        
        # Print scenario costs for debugging
        print("\nDebug scenario costs:")
        overall_setup = 0
        overall_prod = 0
        overall_hold = 0
        overall_total = 0
        
        for sc in scenario_costs:
            print(f"Scenario {sc['scenario']} (p={sc['weight']:.2f}):")
            print(f"  Setup: {sc['setup']:.2f} * {sc['weight']:.2f} = {sc['setup'] * sc['weight']:.2f}")
            print(f"  Production: {sc['production']:.2f} * {sc['weight']:.2f} = {sc['production'] * sc['weight']:.2f}")
            print(f"  Holding: {sc['holding']:.2f} * {sc['weight']:.2f} = {sc['holding'] * sc['weight']:.2f}")
            print(f"  Total: {sc['total']:.2f} * {sc['weight']:.2f} = {sc['weighted']:.2f}")
            
            overall_setup += sc['setup'] * sc['weight']
            overall_prod += sc['production'] * sc['weight']
            overall_hold += sc['holding'] * sc['weight']
            overall_total += sc['weighted']
        
        print(f"\nOverall weighted costs:")
        print(f"  Setup: {overall_setup:.2f}")
        print(f"  Production: {overall_prod:.2f}")
        print(f"  Holding: {overall_hold:.2f}")
        print(f"  Total: {overall_total:.2f}")
        print(f"  Model objective: {self.model.objVal:.2f}")
        
        objective_value = self.model.objVal
        solve_time = time.time() - solve_start
        
        # Store timing statistics
        self.timing_stats = TimingStats(
            total_solve_time=solve_time,
            master_time=solve_time,
            subproblem_time=0,
            ml_prediction_time=0,
            num_iterations=1
        )
        
        # Store solution
        self.best_solution = Solution(
            setup=setup_solution,
            production=production_solution,
            inventory=inventory_solution,
            objective=objective_value
        )
        
        # Print linking period setups for debugging
        linking_setups = {t: setup_solution[0][t] for t in self.params.linking_periods}
        print(f"Direct solver linking period setups: {linking_setups}")
        
        return objective_value, self.timing_stats