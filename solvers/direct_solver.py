import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Tuple

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
    
    def build_model(self):
        """Build the complete optimization model.
        
        Returns:
            Gurobi model
        """
        model = gp.Model("Direct")
        model.setParam('OutputFlag', 0)

        # Create variables for all scenarios and periods
        setup = {}
        production = {}
        inventory = {}
        
        # Decision variables
        for s in range(len(self.params.scenarios)):
            setup[s] = {}
            production[s] = {}
            inventory[s] = {}
            
            for t in range(self.params.total_periods):
                setup[s][t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{s},{t}]")
                production[s][t] = model.addVar(lb=0, name=f"production[{s},{t}]")
                inventory[s][t] = model.addVar(lb=0, name=f"inventory[{s},{t}]")
        
        # ————— Build “here-and-now” variables —————
        here_setup      = {}
        here_production = {}
        for t in self.params.linking_periods:
            # one master setup‐binary per linking period
            here_setup[t] = model.addVar(
                vtype=GRB.BINARY, name=f"here_setup[{t}]"
            )
            # one master production‐continuous per linking period
            here_production[t] = model.addVar(
                lb=0, name=f"here_production[{t}]"
            )
            # link the two so capacity still holds
            model.addConstr(
                here_production[t] <= self.params.capacity * here_setup[t],
                name=f"here_capacity[{t}]"
            )

        # ————— Enforce non-anticipativity —————
        for s in range(len(self.params.scenarios)):
            for t in self.params.linking_periods:
                model.addConstr(
                    setup[s][t]      == here_setup[t],
                    name=f"non_ant_setup[{s},{t}]"
                )
                model.addConstr(
                    production[s][t] == here_production[t],
                    name=f"non_ant_prod[{s},{t}]"
                )

        
        # Flow balance constraints
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            
            # Initial inventory (assume zero initial inventory if not specified)
            initial_inventory = getattr(self.params, 'initial_inventory', 0)
            
            # First period
            model.addConstr(
                inventory[s][0] == initial_inventory - scenario.demands[0] + production[s][0],
                name=f"flow_balance_initial[{s}]"
            )
            
            # Remaining periods
            for t in range(1, self.params.total_periods):
                model.addConstr(
                    inventory[s][t] == inventory[s][t-1] + production[s][t] - scenario.demands[t],
                    name=f"flow_balance[{s},{t}]"
                )
        
        # Production capacity constraints
        for s in range(len(self.params.scenarios)):
            for t in range(self.params.total_periods):
                model.addConstr(
                    production[s][t] <= self.params.capacity * setup[s][t],
                    name=f"capacity[{s},{t}]"
                )
        
        # Optional: Add valid inequalities to strengthen the formulation
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            
            # Minimum number of setups needed
            total_demand = sum(scenario.demands) #make this a max
            min_setups = max(1, int(total_demand // self.params.capacity) + 
                          (1 if total_demand % self.params.capacity > 0 else 0)) 
            
            setup_sum = gp.quicksum(setup[s][t] for t in range(self.params.total_periods))
            model.addConstr(setup_sum >= min_setups, name=f"min_setups[{s}]")
            
            # Ensure sufficient production in each period
            remaining_demand = {}
            remaining_demand[self.params.total_periods - 1] = scenario.demands[self.params.total_periods - 1]
            
            for t in range(self.params.total_periods - 2, -1, -1):
                remaining_demand[t] = remaining_demand[t+1] + scenario.demands[t]
                
                # We need at least this many setups for the remaining demand
                remaining_min_setups = max(1, int(remaining_demand[t] // self.params.capacity) + 
                                        (1 if remaining_demand[t] % self.params.capacity > 0 else 0))
                
                remaining_setup_sum = gp.quicksum(setup[s][i] for i in range(t, self.params.total_periods))
                model.addConstr(remaining_setup_sum >= remaining_min_setups, 
                              name=f"remaining_min_setups[{s},{t}]")
        
        # Objective function: minimize expected total cost
        obj = gp.LinExpr()
        
        for s in range(len(self.params.scenarios)):
            scenario = self.params.scenarios[s]
            
            # Setup costs
            setup_cost = gp.quicksum(self.params.fixed_cost * setup[s][t] 
                                  for t in range(self.params.total_periods))
            
            # Production costs
            prod_cost = gp.quicksum(self.params.production_cost * production[s][t] 
                                  for t in range(self.params.total_periods))
            
            # Inventory holding costs (use default value of 0 if holding_cost not defined)
            holding_cost = getattr(self.params, 'holding_cost', 0)
            inv_cost = gp.quicksum(holding_cost * inventory[s][t] 
                                for t in range(self.params.total_periods))
            
            # Total cost for this scenario
            scenario_cost = setup_cost + prod_cost + inv_cost
            
            # Add to expected cost
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
        
        for s in range(len(self.params.scenarios)):
            setup_s = []
            production_s = []
            inventory_s = []
            
            for t in range(self.params.total_periods):
                setup_var = self.model.getVarByName(f"setup[{s},{t}]")
                production_var = self.model.getVarByName(f"production[{s},{t}]")
                inventory_var = self.model.getVarByName(f"inventory[{s},{t}]")
                
                setup_s.append(setup_var.x > 0.5)
                production_s.append(production_var.x)
                inventory_s.append(inventory_var.x)
            
            setup_solution.append(setup_s)
            production_solution.append(production_s)
            inventory_solution.append(inventory_s)
        
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
        
        return objective_value, self.timing_stats