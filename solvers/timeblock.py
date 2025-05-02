from typing import Dict, Tuple, Set, List, Optional
import gurobipy as gp
from gurobipy import GRB
import time
import torch
import numpy as np

from data.data_structures import ScenarioData, ProblemParameters


class MLTimeBlockSubproblem:
    """Time block subproblem solver with enhanced ML capability for multiple candidate solutions."""

    def __init__(self, start_period: int, num_periods: int,
                 scenario_data: ScenarioData, params: ProblemParameters,
                 ml_predictor=None):
        """Initialize the time block subproblem."""
        self.start_period = start_period
        self.num_periods = num_periods
        self.scenario_data = scenario_data
        self.params = params
        self.ml_predictor = ml_predictor
        self.model = None
        self.setup_vars = {}
        self.prod_vars = {}
        self.inv_vars = {}
        self.ml_solution = None  # Store the solution predicted by ML
        self.tried_solutions = []  # Keep track of all solutions we've tried
        self.ml_prediction_time = 0.0  # Track ML prediction time

    def predict_solution(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """
        Use ML to predict solutions with top-K capability instead of solving with Gurobi.

        Args:
            initial_inventory: Initial inventory level for this block.
            fixed_setups: Dictionary of setup decisions fixed by the master problem.

        Returns:
            Tuple of (cost, is_feasible): Best predicted cost and feasibility status.
        """
        if self.ml_predictor is None:
            return float('inf'), False

        if not self.ml_predictor.is_loaded or self.ml_predictor.model is None:
             print(f"ML predictor not loaded or model is None for block starting {self.start_period}.")
             return float('inf'), False

        try:
            ml_start_time = time.time()
            
            # Prepare features for ML prediction
            features = self.ml_predictor.prepare_features(
                fixed_setups=fixed_setups,
                scenario=self.scenario_data,
                params=self.params,
                start_period=self.start_period,
                initial_inventory=initial_inventory
            )

            if features is None:
                 print(f"Feature preparation failed for block starting at {self.start_period}.")
                 return float('inf'), False

            # Get multiple candidate solutions
            ml_solutions = self.ml_predictor.predict_top_k_solutions(features, self.num_periods)
            
            if not ml_solutions:
                print(f"ML prediction returned no solutions for block starting at {self.start_period}.")
                return float('inf'), False
                
            # Clear previous tried solutions
            self.tried_solutions = []
                
            # Try each solution, keeping track of the best
            best_cost = float('inf')
            best_solution = None
            is_any_feasible = False
            
            for solution_idx, (setup_pred_binary, production_plan, inventory_plan) in enumerate(ml_solutions):
                # Post-process and correct this solution candidate
                corrected_setup = list(setup_pred_binary)
                corrected_production = list(production_plan)
                corrected_inventory = [0.0] * (self.num_periods + 1)
                corrected_inventory[0] = initial_inventory

                # Apply master decisions and correct inconsistencies
                for t in range(self.num_periods):
                    period_in_full_horizon = self.start_period + t

                    # 1. Respect Master Setups for Linking Periods
                    if period_in_full_horizon in self.params.linking_periods:
                        master_decision = fixed_setups.get(period_in_full_horizon, False)
                        if bool(corrected_setup[t]) != master_decision:
                            corrected_setup[t] = float(master_decision)

                    # 2. Ensure Production matches Setup state
                    if corrected_production[t] > 1e-6 and not corrected_setup[t]:
                        corrected_production[t] = 0.0
                    elif corrected_setup[t] and corrected_production[t] < 0:
                         corrected_production[t] = 0.0

                    # 3. Ensure Production respects Capacity
                    capacity_limit = self.params.capacity * corrected_setup[t]
                    if corrected_production[t] > capacity_limit + 1e-6:
                        corrected_production[t] = capacity_limit

                    # 4. Recalculate Inventory based on corrected values
                    demand = self.scenario_data.demands[period_in_full_horizon] if period_in_full_horizon < len(self.scenario_data.demands) else 0
                    corrected_inventory[t+1] = max(0.0, corrected_inventory[t] + corrected_production[t] - demand)

                # Store this corrected solution
                current_solution = {
                    'setup': [bool(s) for s in corrected_setup],
                    'production': corrected_production.copy(),
                    'inventory': corrected_inventory.copy()
                }
                
                # Add to our list of tried solutions
                self.tried_solutions.append(current_solution)

                # Calculate cost based on this corrected solution
                cost = 0.0
                for t in range(self.num_periods):
                    if current_solution['setup'][t]:
                        cost += self.params.fixed_cost
                    cost += current_solution['production'][t] * self.params.production_cost
                    cost += current_solution['inventory'][t+1] * self.params.holding_cost

                # Verify feasibility of this solution
                is_feasible = self.verify_solution_feasibility(
                    initial_inventory, fixed_setups,
                    current_solution['setup'],
                    current_solution['production'],
                    current_solution['inventory'][1:]
                )

                if is_feasible and cost < best_cost:
                    best_cost = cost
                    best_solution = current_solution
                    is_any_feasible = True

            # Calculate total ML time
            self.ml_prediction_time = time.time() - ml_start_time
                    
            # If we found at least one feasible solution
            if is_any_feasible:
                self.ml_solution = best_solution
                return best_cost, True
            else:
                print(f"None of the {len(ml_solutions)} ML solutions were feasible for block {self.start_period}.")
                self.ml_solution = None
                return float('inf'), False

        except Exception as e:
            print(f"ML prediction/correction error in block {self.start_period}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.ml_solution = None
            return float('inf'), False

    def verify_solution_feasibility(self, initial_inv, fixed_setups, setup_list, production_list, inventory_end_of_period_list):
        """Verify if the ML-predicted solution is feasible."""
        if len(inventory_end_of_period_list) != self.num_periods:
             return False

        current_inv = initial_inv
        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            setup_t = setup_list[t]
            production_t = production_list[t]
            inventory_end_t = inventory_end_of_period_list[t]

            # Check Production vs Setup/Capacity
            if production_t < -1e-6:  # Negative production
                 return False
            if production_t > self.params.capacity * setup_t + 1e-6:
                 return False
            if production_t > 1e-6 and not setup_t:
                 return False

            # Check Master Setup Consistency
            if period_in_full_horizon in self.params.linking_periods:
                master_setup = fixed_setups.get(period_in_full_horizon, False)
                if setup_t != master_setup:
                     return False

            # Check Flow Balance and Inventory Non-Negativity
            demand = self.scenario_data.demands[period_in_full_horizon] if period_in_full_horizon < len(self.scenario_data.demands) else 0
            calculated_end_inv = current_inv + production_t - demand

            if calculated_end_inv < -1e-6:  # Check if calculated inventory goes negative
                 return False
            if abs(calculated_end_inv - inventory_end_t) > 1e-4:  # Check if provided inventory matches calculation
                 return False

            # Update inventory for the next iteration
            current_inv = inventory_end_t

        # If all checks pass
        return True

    def build_model(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """Build Gurobi model as fallback if ML prediction fails or is infeasible."""
        model = gp.Model(f"TimeBlock_{self.start_period}")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)

        # Variables
        self.setup_vars = {}
        self.prod_vars = {}
        self.inv_vars = {}

        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            # Setup variable
            self.setup_vars[t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{t}]")

            # Fix setup if it's a linking period
            if period_in_full_horizon in self.params.linking_periods:
                master_decision = fixed_setups.get(period_in_full_horizon, False)
                self.setup_vars[t].lb = float(master_decision)
                self.setup_vars[t].ub = float(master_decision)

            # Production variable
            self.prod_vars[t] = model.addVar(lb=0.0, name=f"production[{t}]")

            # Inventory variable
            self.inv_vars[t] = model.addVar(lb=0.0, name=f"inventory[{t}]")

        # Constraints
        current_inv_var = initial_inventory
        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            demand = self.scenario_data.demands[period_in_full_horizon] if period_in_full_horizon < len(self.scenario_data.demands) else 0

            # Inventory balance
            model.addConstr(
                self.inv_vars[t] == current_inv_var + self.prod_vars[t] - demand,
                name=f"balance_{t}"
            )

            # Capacity constraint
            model.addConstr(
                self.prod_vars[t] <= self.params.capacity * self.setup_vars[t],
                name=f"capacity_{t}"
            )

            # Update inventory variable
            current_inv_var = self.inv_vars[t]

        # Objective: Minimize costs within this block
        obj = gp.quicksum(
            self.params.fixed_cost * self.setup_vars[t] +
            self.params.production_cost * self.prod_vars[t] +
            self.params.holding_cost * self.inv_vars[t]
            for t in range(self.num_periods)
        )
        model.setObjective(obj, GRB.MINIMIZE)

        self.model = model
        return model

    def solve(self) -> Tuple[float, bool]:
        """Solve using Gurobi (fallback)."""
        if self.model is None:
             print(f"Error: Gurobi model not built for block starting {self.start_period}.")
             return float('inf'), False
        try:
            self.model.optimize()

            if self.model.status == GRB.OPTIMAL:
                return self.model.objVal, True
            elif self.model.status == GRB.INFEASIBLE:
                print(f"Warning: Gurobi subproblem (block starting {self.start_period}) is infeasible.")
                return float('inf'), False
            else:
                print(f"Warning: Gurobi subproblem (block starting {self.start_period}) ended with status {self.model.status}.")
                return float('inf'), False
        except gp.GurobiError as e:
            print(f"Gurobi error solving block starting {self.start_period}: {e}")
            return float('inf'), False

    def get_solution(self) -> Dict:
        """Get solution (either from ML or Gurobi)."""
        # Prioritize ML solution if available
        if self.ml_solution is not None:
             return self.ml_solution.copy()

        # Fall back to Gurobi solution
        if self.model is not None and hasattr(self.model, 'status') and self.model.status == GRB.OPTIMAL:
            inventory_solution = [self.inv_vars[t].x for t in range(self.num_periods)]
            last_period_idx = self.num_periods - 1
            final_inventory = self.inv_vars[last_period_idx].x if last_period_idx >= 0 else 0.0

            return {
                'setup': [self.setup_vars[t].x > 0.5 for t in range(self.num_periods)],
                'production': [self.prod_vars[t].x for t in range(self.num_periods)],
                'inventory': inventory_solution + [final_inventory]
            }
        else:
            print(f"Warning: No valid solution found for block starting {self.start_period}.")
            return {
                'setup': [False] * self.num_periods,
                'production': [0.0] * self.num_periods,
                'inventory': [0.0] * (self.num_periods + 1)
            }

    def get_critical_setups(self) -> Set[int]:
        """Identify critical setup periods from the obtained solution."""
        critical_periods = set()
        sol = self.get_solution()
        
        if not sol or not sol['setup']:
             return critical_periods

        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            if period_in_full_horizon not in self.params.linking_periods:
                if t < len(sol['setup']) and sol['setup'][t]:
                    if t < len(sol['production']) and sol['production'][t] > 0.01 * self.params.capacity:
                        critical_periods.add(period_in_full_horizon)

        return critical_periods
        
    def get_all_tried_solutions(self) -> List[Dict]:
        """Return all solutions that were tried by the ML predictor.
        
        Returns:
            List of dictionaries containing setup, production, and inventory plans
        """
        return self.tried_solutions.copy() if self.tried_solutions else []


class HybridMLTimeBlockSubproblem:
    """
    Hybrid time block subproblem solver that:
    1. Uses ML to predict multiple candidate setups (binary variables)
    2. Solves the LP with fixed setups for optimal production and inventory
    3. Returns the best feasible solution among all candidates
    """

    def __init__(self, start_period: int, num_periods: int,
                 scenario_data: ScenarioData, params: ProblemParameters,
                 ml_predictor=None):
        """Initialize the hybrid time block subproblem."""
        self.start_period = start_period
        self.num_periods = num_periods
        self.scenario_data = scenario_data
        self.params = params
        self.ml_predictor = ml_predictor
        self.model = None  # For the full MIP fallback
        self.setup_vars = {}  # For the full MIP fallback
        self.prod_vars = {}  # For the full MIP fallback
        self.inv_vars = {}  # For the full MIP fallback
        self.hybrid_solution = None  # Stores the best hybrid solution
        self.tried_solutions = []  # Keep track of all candidate solutions
        self.ml_prediction_time = 0.0  # Track ML prediction time
        self.lp_solution_time = 0.0  # Track LP solution time

    def predict_setups(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """Use ML to predict multiple setup decision candidates.
        
        Returns:
            List of setup decision lists, or None if prediction failed
        """
        if self.ml_predictor is None or not self.ml_predictor.is_loaded:
            print(f"Hybrid: ML predictor not available for block {self.start_period}.")
            return None

        try:
            ml_start_time = time.time()
            
            # Prepare features
            features = self.ml_predictor.prepare_features(
                fixed_setups=fixed_setups,
                scenario=self.scenario_data,
                params=self.params,
                start_period=self.start_period,
                initial_inventory=initial_inventory
            )

            if features is None:
                 print(f"Hybrid: Feature preparation failed for block {self.start_period}.")
                 return None

            # Get multiple setup predictions
            ml_solutions = self.ml_predictor.predict_top_k_solutions(features, self.num_periods)
            
            if not ml_solutions:
                print(f"Hybrid: ML prediction returned no solutions for block {self.start_period}.")
                return None
            
            # Extract just the setup decisions
            setup_candidates = []
            for setup_pred, _, _ in ml_solutions:
                # Ensure setups respect master's fixed decisions
                final_setup = []
                for t in range(self.num_periods):
                    period_in_full_horizon = self.start_period + t
                    if period_in_full_horizon in self.params.linking_periods:
                        final_setup.append(fixed_setups.get(period_in_full_horizon, False))
                    else:
                        if t < len(setup_pred):
                            final_setup.append(bool(setup_pred[t]))
                        else:
                            final_setup.append(False)
                            
                setup_candidates.append(final_setup)
            
            self.ml_prediction_time = time.time() - ml_start_time
            return setup_candidates

        except Exception as e:
            print(f"Hybrid: ML setup prediction error in block {self.start_period}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def solve_with_fixed_setups(self, initial_inventory: float, setup_decisions: List[bool]):
        """Solve the continuous subproblem (LP) with fixed setup decisions."""
        lp_start_time = time.time()
        
        model = gp.Model(f"ContinuousSubproblem_{self.start_period}")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)

        # Production and inventory variables only
        prod_vars = {}
        inv_vars = {}  # inventory[t] is end of period t

        for t in range(self.num_periods):
            prod_vars[t] = model.addVar(lb=0.0, name=f"production[{t}]")
            inv_vars[t] = model.addVar(lb=0.0, name=f"inventory[{t}]")

        # Constraints
        current_inv_var = initial_inventory
        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            demand = self.scenario_data.demands[period_in_full_horizon] if period_in_full_horizon < len(self.scenario_data.demands) else 0

            # Inventory balance: end_inv[t] == start_inv[t] + prod[t] - demand
            model.addConstr(
                inv_vars[t] == current_inv_var + prod_vars[t] - demand,
                name=f"balance_{t}"
            )

            # Capacity constraint with fixed setup decision
            model.addConstr(
                prod_vars[t] <= self.params.capacity * setup_decisions[t],
                name=f"capacity_{t}"
            )

            # Update inventory variable for the next period
            current_inv_var = inv_vars[t]

        # Objective - only includes production and inventory costs
        obj = gp.quicksum(
            self.params.production_cost * prod_vars[t] +
            self.params.holding_cost * inv_vars[t]  # Cost on ending inventory
            for t in range(self.num_periods)
        )

        model.setObjective(obj, GRB.MINIMIZE)
        try:
            model.optimize()

            # Update LP solution time
            self.lp_solution_time = time.time() - lp_start_time

            if model.status == GRB.OPTIMAL:
                # Extract solution
                production = [prod_vars[t].x for t in range(self.num_periods)]
                # Get final inventory state
                inventory = [inv_vars[t].x for t in range(self.num_periods)]
                
                # Add final inventory state
                final_inventory = inventory[-1] if inventory else initial_inventory
                inventory.append(final_inventory)

                # Calculate total cost including fixed setup costs
                setup_cost = sum(self.params.fixed_cost for t in range(self.num_periods)
                                if setup_decisions[t])
                total_cost = model.objVal + setup_cost

                solution = {
                    'setup': setup_decisions,  # The fixed setups used
                    'production': production,
                    'inventory': inventory  # Includes final state
                }

                return total_cost, True, solution
            else:
                if model.status == GRB.INFEASIBLE:
                    # Could compute and save IIS for debugging
                    pass
                return float('inf'), False, None
        except gp.GurobiError as e:
            print(f"Gurobi error solving continuous subproblem for block {self.start_period}: {e}")
            return float('inf'), False, None

    def solve_hybrid(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """
        Solve using hybrid approach: 
        1. ML for multiple setup candidates
        2. LP for continuous variables with each setup candidate
        3. Return the best feasible solution
        """
        # 1. Use ML to predict multiple setup decision candidates
        setup_candidates = self.predict_setups(initial_inventory, fixed_setups)

        if not setup_candidates:
            # ML prediction failed, fall back to full MIP
            print(f"Hybrid: ML prediction failed for block {self.start_period}, falling back to MIP.")
            return self.solve_traditional(initial_inventory, fixed_setups)

        # Clear previous solutions
        self.tried_solutions = []
        
        # 2. Try each setup candidate with LP solver
        best_cost = float('inf')
        best_solution = None
        is_any_feasible = False
        
        for idx, setup_decisions in enumerate(setup_candidates):
            # Solve continuous problem with this setup configuration
            cost, is_feasible, solution = self.solve_with_fixed_setups(
                initial_inventory, setup_decisions)
                
            if solution:
                self.tried_solutions.append(solution)
                
            if is_feasible and cost < best_cost:
                best_cost = cost
                best_solution = solution
                is_any_feasible = True
                
            # Debug output
            # print(f"Hybrid candidate {idx+1}/{len(setup_candidates)} - Feasible: {is_feasible}, Cost: {cost:.2f}")

        if is_any_feasible:
            # Store best hybrid solution and return result
            self.hybrid_solution = best_solution
            return best_cost, True
        else:
            # No feasible solution with any candidate, fall back to MIP
            print(f"Hybrid: No feasible LP solution for block {self.start_period} with any ML setup, falling back to MIP.")
            return self.solve_traditional(initial_inventory, fixed_setups)

    def solve_traditional(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """Fall back to traditional MIP solution for the block."""
        self.build_mip_model(initial_inventory, fixed_setups)
        return self.solve_mip()

    def build_mip_model(self, initial_inventory: float, fixed_setups: Dict[int, bool]):
        """Build Gurobi MIP model as fallback."""
        model = gp.Model(f"MIPTimeBlock_{self.start_period}")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 1)

        self.setup_vars = {}
        self.prod_vars = {}
        self.inv_vars = {}

        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            self.setup_vars[t] = model.addVar(vtype=GRB.BINARY, name=f"setup[{t}]")
            if period_in_full_horizon in self.params.linking_periods:
                master_decision = fixed_setups.get(period_in_full_horizon, False)
                self.setup_vars[t].lb = float(master_decision)
                self.setup_vars[t].ub = float(master_decision)
            self.prod_vars[t] = model.addVar(lb=0.0, name=f"production[{t}]")
            self.inv_vars[t] = model.addVar(lb=0.0, name=f"inventory[{t}]")

        current_inv_var = initial_inventory
        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            demand = self.scenario_data.demands[period_in_full_horizon] if period_in_full_horizon < len(self.scenario_data.demands) else 0
            model.addConstr(self.inv_vars[t] == current_inv_var + self.prod_vars[t] - demand, name=f"balance_{t}")
            model.addConstr(self.prod_vars[t] <= self.params.capacity * self.setup_vars[t], name=f"capacity_{t}")
            current_inv_var = self.inv_vars[t]

        obj = gp.quicksum(
            self.params.fixed_cost * self.setup_vars[t] +
            self.params.production_cost * self.prod_vars[t] +
            self.params.holding_cost * self.inv_vars[t]
            for t in range(self.num_periods)
        )
        model.setObjective(obj, GRB.MINIMIZE)
        self.model = model
        return model

    def solve_mip(self) -> Tuple[float, bool]:
        """Solve using Gurobi MIP (fallback)."""
        if self.model is None:
             print(f"Error: MIP model not built for block {self.start_period}.")
             return float('inf'), False
        try:
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                return self.model.objVal, True
            elif self.model.status == GRB.INFEASIBLE:
                 print(f"Warning: Fallback MIP (block {self.start_period}) is infeasible.")
                 return float('inf'), False
            else:
                print(f"Warning: Fallback MIP (block {self.start_period}) ended status {self.model.status}.")
                return float('inf'), False
        except gp.GurobiError as e:
            print(f"Gurobi error solving fallback MIP for block {self.start_period}: {e}")
            return float('inf'), False

    def get_solution(self) -> Dict:
        """Get solution (prioritize hybrid, fallback to MIP)."""
        if self.hybrid_solution is not None:
            return self.hybrid_solution.copy()

        # Fall back to MIP solution if available
        if self.model is not None and hasattr(self.model, 'status') and self.model.status == GRB.OPTIMAL:
             inventory_solution = [self.inv_vars[t].x for t in range(self.num_periods)]
             final_inventory = self.inv_vars[self.num_periods-1].x if self.num_periods > 0 else 0.0

             return {
                 'setup': [self.setup_vars[t].x > 0.5 for t in range(self.num_periods)],
                 'production': [self.prod_vars[t].x for t in range(self.num_periods)],
                 'inventory': inventory_solution + [final_inventory]
             }
        else:
            print(f"Warning: No valid solution (hybrid or MIP) found for block {self.start_period}.")
            # Return default structure
            return {
                'setup': [False] * self.num_periods,
                'production': [0.0] * self.num_periods,
                'inventory': [0.0] * (self.num_periods + 1)
            }

    def get_critical_setups(self) -> Set[int]:
        """Identify critical setup periods from the obtained solution (hybrid or MIP)."""
        critical_periods = set()
        sol = self.get_solution()
        if not sol or not sol['setup']:
            return critical_periods

        for t in range(self.num_periods):
            period_in_full_horizon = self.start_period + t
            if period_in_full_horizon not in self.params.linking_periods:
                if t < len(sol['setup']) and sol['setup'][t]:
                    # Use a small threshold relative to capacity
                    if t < len(sol['production']) and sol['production'][t] > 0.01 * self.params.capacity:
                        critical_periods.add(period_in_full_horizon)
        return critical_periods
        
    def get_all_tried_solutions(self) -> List[Dict]:
        """Return all solutions that were tried by the hybrid solver.
        
        Returns:
            List of dictionaries containing setup, production, and inventory plans
        """
        return self.tried_solutions.copy() if self.tried_solutions else []