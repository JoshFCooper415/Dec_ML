import time
from typing import Dict, Tuple, List, Optional, Set
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from data.data_structures import ProblemParameters, TimingStats, Solution
from models.ml_predictor import MLSubproblemPredictor
from solvers.scenario import MLScenarioSubproblem, HybridScenarioSubproblem
from solvers.laporte_cuts import LaporteCutsGenerator  # Import Laporte cuts generator


class MLBendersDecomposition:
    """Enhanced Benders decomposition with top-K ML prediction support."""
    
    def __init__(self,
             params: ProblemParameters,
             use_ml: bool = True,
             use_hybrid: bool = True,
             use_trust_region: bool = False,
             use_valid_cuts: bool = True,
             use_laporte_cuts: bool = False,
             ml_candidates: int = 10):
        """Initialize the improved Benders decomposition solver.
        
        Args:
            params: Problem parameters
            use_ml: Whether to use ML prediction
            use_hybrid: Whether to use hybrid approach (ML for binary, MIP for continuous)
            use_trust_region: Whether to enforce a trust region on master solutions
            use_valid_cuts: Whether to use mathematically valid cuts (LP relaxation)
            use_laporte_cuts: Whether to generate Laporte cuts
            ml_candidates: Number of alternative ML solutions to generate (top-K)
        """
        self.params = params
        self.use_ml = use_ml
        self.use_hybrid = use_hybrid and use_ml  # Hybrid requires ML
        self.use_trust_region = use_trust_region
        self.use_valid_cuts = use_valid_cuts
        self.use_laporte_cuts = use_laporte_cuts
        
        # Initialize ML predictor with top-K capability
        self.ml_predictor = None
        if use_ml:
            self.ml_predictor = MLSubproblemPredictor(k=ml_candidates)
            print(f"ML predictor initialized with {ml_candidates} candidate solutions")

        # Use the appropriate subproblem class based on the approach
        if self.use_hybrid:
            self.scenario_problems = [
                HybridScenarioSubproblem(params.scenarios[idx], params, self.ml_predictor)
                for idx in range(len(params.scenarios))
            ]
            print(f"Using hybrid approach with {len(params.scenarios)} scenario problems")
        else:
            self.scenario_problems = [
                MLScenarioSubproblem(params.scenarios[idx], params, self.ml_predictor)
                for idx in range(len(params.scenarios))
            ]
            print(f"Using standard ML approach with {len(params.scenarios)} scenario problems")

        # Laporte cuts helper
        self.laporte_cuts_generator = (
            LaporteCutsGenerator(params)
            if use_laporte_cuts else None
        )

        # Master model and solution storage
        self.master_model = None
        self.best_solution = None
        self.timing_stats = None

        # Here-and-now variables
        self.here_setup = {}
        self.here_production = {}

        # For trust-region, keep last few binary vectors
        self.previous_master_solutions = []
        self.max_saved_solutions = 5

        # Track constraints
        self.trust_region_constrs = []
        self.feasibility_cuts = []
        self.optimality_cuts = []
        self.laporte_cuts = []
        
        # Cut counters
        self.num_feasibility_cuts = 0
        self.num_valid_optimality_cuts = 0
        self.num_heuristic_optimality_cuts = 0
        self.num_trust_region_cuts = 0
        self.num_laporte_cuts = 0
        
        # ML statistics
        self.ml_success_rate = 0.0
        self.ml_prediction_time = 0.0
        self.ml_attempts = 0
        self.ml_successes = 0

    def build_master_problem(self) -> gp.Model:
        """Construct the master problem."""
        model = gp.Model("MasterBenders")
        model.setParam('OutputFlag', 0)
        
        # Set LazyConstraints parameter to 1 to enable lazy constraints
        model.setParam('LazyConstraints', 1)

        # ------ Here-and-now decisions for linking periods ------
        self.here_setup.clear()
        self.here_production.clear()
        for t in self.params.linking_periods:
            self.here_setup[t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"setup[{t}]"
            )
            self.here_production[t] = model.addVar(
                lb=0,
                name=f"production[{t}]"
            )
            # capacity enforcement in master
            model.addConstr(
                self.here_production[t]
                <= self.params.capacity * self.here_setup[t],
                name=f"here_capacity[{t}]"
            )

        # Theta vars approximate recourse cost per scenario
        theta = {}
        for s in range(len(self.params.scenarios)):
            theta[s] = model.addVar(lb=0, name=f"theta[{s}]")

        # Initial trivial LB cuts on theta
        for s_idx, scenario in enumerate(self.params.scenarios):
            total_demand = sum(scenario.demands)
            # at least capacity covers demand
            min_setups = max(
                1,
                (total_demand + self.params.capacity - 1) // self.params.capacity
            )
            model.addConstr(
                theta[s_idx]
                >= min_setups * self.params.fixed_cost
                + total_demand * self.params.production_cost,
                name=f"initial_cut[{s_idx}]"
            )

        # Objective: expected theta
        objective = gp.quicksum(
            scenario.probability * theta[s]
            for s, scenario in enumerate(self.params.scenarios)
        )
        model.setObjective(objective, GRB.MINIMIZE)

        # Update model to make variables/constraints available for reference
        model.update()
        
        self.master_model = model
        return model

    def update_trust_region_constraint(self, iteration: int) -> bool:
        """Add or update trust region constraint."""
        if not self.use_trust_region or iteration == 0:
            return False
        if not self.previous_master_solutions:
            return False

        # Remove old trust-region constraints
        for old_constr in self.trust_region_constrs:
            if old_constr in self.master_model.getConstrs():
                self.master_model.remove(old_constr)
        self.trust_region_constrs = []

        prev = self.previous_master_solutions[-1]
        diff = gp.LinExpr()
        
        for t in self.params.linking_periods:
            if t in prev:  # Ensure t is in prev before accessing
                var = self.master_model.getVarByName(f"setup[{t}]")
                if var is not None:  # Only add if variable exists
                    diff += (1 - var) if prev[t] else var

        base_radius = max(1, len(self.params.linking_periods) // 10)
        growth = min(iteration // 3, 3)
        max_radius = len(self.params.linking_periods) // 3
        radius = min(base_radius + growth, max_radius)

        # Only add constraint if we have a valid expression
        if diff.size() > 0:  # Check if expression has terms
            tr_constr = self.master_model.addConstr(
                diff <= radius,
                name=f"trust_region_{iteration}"
            )
            # Mark as lazy constraint by setting the Lazy attribute
            tr_constr.Lazy = 1
            self.trust_region_constrs.append(tr_constr)
            self.num_trust_region_cuts += 1
            return True
        return False

    def extract_master_solution(self) -> Tuple[Dict[int, bool], Dict[int, float]]:
        """Retrieve here-and-now decisions for all linking periods."""
        setups = {}
        prods = {}
        for t in self.params.linking_periods:
            setup_var = self.master_model.getVarByName(f"setup[{t}]")
            prod_var = self.master_model.getVarByName(f"production[{t}]")
            
            # Include only if variables exist
            if setup_var is not None:
                setups[t] = (setup_var.x > 0.5)
            if prod_var is not None:
                prods[t] = prod_var.x

        # record for trust region
        self.previous_master_solutions.append(setups.copy())
        if len(self.previous_master_solutions) > self.max_saved_solutions:
            self.previous_master_solutions.pop(0)

        return setups, prods

    def add_feasibility_cut(self, scenario_idx: int, dual_rays: List[Tuple[List[float], float]]) -> bool:
        """Add a feasibility cut based on extreme rays of the dual subproblem.
        
        Args:
            scenario_idx: Index of the scenario
            dual_rays: List of extreme rays from dual subproblem (coefficients, constant)
            
        Returns:
            True if a cut was added, False otherwise
        """
        theta = self.master_model.getVarByName(f"theta[{scenario_idx}]")
        if theta is None:
            return False
            
        for ray_coeffs, ray_const in dual_rays:
            # Build cut expression
            expr = gp.LinExpr(ray_const)
            
            # Add terms for each complicating variable (setups)
            for t_idx, t in enumerate(self.params.linking_periods):
                if t_idx < len(ray_coeffs):
                    var = self.master_model.getVarByName(f"setup[{t}]")
                    if var is not None and abs(ray_coeffs[t_idx]) > 1e-10:
                        expr.addTerms(ray_coeffs[t_idx], var)
            
            # Add the feasibility cut: ray_const + sum(ray_coeffs[i] * y[i]) <= 0
            constr = self.master_model.addConstr(
                expr <= 0,
                name=f"feasibility_cut_{self.num_feasibility_cuts}"
            )
            
            # Set the Lazy attribute to mark as a lazy constraint
            constr.Lazy = 1
            self.feasibility_cuts.append(constr)
            self.num_feasibility_cuts += 1
        
        return True

    def add_valid_optimality_cut(self, scenario_idx: int, dual_values: Dict, objective_value: float) -> bool:
        """Add a valid optimality cut based on the LP relaxation dual solution.
        
        Args:
            scenario_idx: Index of the scenario
            dual_values: Dictionary of dual values for the constraints in the LP relaxation
            objective_value: Optimal objective value of the subproblem
            
        Returns:
            True if a cut was added, False otherwise
        """
        theta = self.master_model.getVarByName(f"theta[{scenario_idx}]")
        if theta is None:
            return False
        
        # Calculate cut coefficients based on the dual values
        cut_expr = gp.LinExpr()
        rhs = objective_value
        
        # Add terms for linking periods
        for t in self.params.linking_periods:
            # Get the capacity constraint dual value (if available)
            capacity_dual_key = f"capacity_dual_{t}"
            if capacity_dual_key in dual_values:
                capacity_dual = dual_values[capacity_dual_key]
                var = self.master_model.getVarByName(f"setup[{t}]")
                if var is not None and abs(capacity_dual) > 1e-10:
                    # For each binary variable, add its coefficient
                    cut_expr.addTerms(capacity_dual * self.params.capacity, var)
        
        # Add the optimality cut: theta >= rhs + cut_expr
        constr = self.master_model.addConstr(
            theta >= rhs + cut_expr,
            name=f"valid_optimality_cut_{self.num_valid_optimality_cuts}"
        )
        
        # Set the Lazy attribute to mark as a lazy constraint
        constr.Lazy = 1
        self.optimality_cuts.append(constr)
        self.num_valid_optimality_cuts += 1
        
        return True
        
    def add_heuristic_optimality_cut(self, scenario_idx: int, objective_value: float, 
                                    current_setups: Dict[int, bool]) -> bool:
        """Add a heuristic optimality cut (used only when valid cuts are disabled).
        
        Args:
            scenario_idx: Index of the scenario
            objective_value: Optimal objective value of the subproblem
            current_setups: Current setup decisions
            
        Returns:
            True if a cut was added, False otherwise
        """
        # Don't add heuristic cuts if valid cuts are enabled
        if self.use_valid_cuts:
            return False
            
        theta = self.master_model.getVarByName(f"theta[{scenario_idx}]")
        if theta is None:
            return False
        
        # Create no-good style optimality cut
        expr = gp.LinExpr()
        total = 0
        
        for t in self.params.linking_periods:
            if t in current_setups:
                var = self.master_model.getVarByName(f"setup[{t}]")
                if var is not None:
                    expr += var if current_setups[t] else (1 - var)
                    total += 1
        
        if total == 0:
            return False
            
        # Create the cut: theta >= objective_value * (sum_expr / total)
        # This ensures that when all setup decisions match current_setups, theta must be >= objective_value
        constr = self.master_model.addConstr(
            theta >= objective_value * (expr / total),
            name=f"heuristic_optimality_cut_{self.num_heuristic_optimality_cuts}"
        )
        
        # Set the Lazy attribute to mark as a lazy constraint
        constr.Lazy = 1
        self.optimality_cuts.append(constr)
        self.num_heuristic_optimality_cuts += 1
        
        return True

    def solve_lp_relaxation(self, scenario_idx: int, fixed_setups: Dict[int, bool]) -> Tuple[float, Dict, List[Tuple[List[float], float]], Optional[Dict]]:
        """Solve the LP relaxation of the subproblem with fixed setups.
        
        Args:
            scenario_idx: Index of the scenario
            fixed_setups: Dictionary of setup decisions fixed by the master problem
            
        Returns:
            Tuple of (objective_value, dual_values, dual_rays, primal_solution)
            - objective_value: Optimal objective value or infinity if infeasible
            - dual_values: Dictionary of dual values for constraints
            - dual_rays: List of extreme rays from dual subproblem (empty if feasible)
            - primal_solution: Dictionary of primal solution (None if infeasible)
        """
        # Get scenario data
        scenario = self.params.scenarios[scenario_idx]
        
        # Create relaxed LP model for this scenario with fixed setups
        model = gp.Model("RelaxedSubproblem")
        model.setParam('OutputFlag', 0)
        
        # Variables with relaxed integrality
        setup = {}
        production = {}
        inventory = {}
        
        # Initialize with 0 inventory
        initial_inventory = 0.0
        
        # Add variables and constraints
        for t in range(self.params.total_periods):
            # Setup variable - relaxed to continuous between 0 and 1
            if t in self.params.linking_periods:
                # Fixed by master problem
                setup[t] = model.addVar(lb=0, ub=1.0, name=f"setup[{t}]")
                setup[t].lb = setup[t].ub = 1.0 if fixed_setups.get(t, False) else 0.0
            else:
                # Free to decide - relaxed to continuous
                setup[t] = model.addVar(lb=0, ub=1.0, name=f"setup[{t}]")
            
            # Production and inventory variables
            production[t] = model.addVar(lb=0, name=f"production[{t}]")
            inventory[t] = model.addVar(lb=0, name=f"inventory[{t}]")
            
            # Capacity constraint
            model.addConstr(
                production[t] <= self.params.capacity * setup[t],
                name=f"capacity[{t}]"
            )
        
        # Flow balance constraints
        for t in range(self.params.total_periods):
            if t == 0:
                # First period
                model.addConstr(
                    inventory[0] == initial_inventory + production[0] - scenario.demands[0],
                    name=f"flow_balance[0]"
                )
            else:
                # Remaining periods
                model.addConstr(
                    inventory[t] == inventory[t-1] + production[t] - scenario.demands[t],
                    name=f"flow_balance[{t}]"
                )
        
        # Objective: minimize total cost
        obj = gp.quicksum(
            self.params.fixed_cost * setup[t] + 
            self.params.production_cost * production[t] + 
            self.params.holding_cost * inventory[t]
            for t in range(self.params.total_periods)
        )
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Solve the model
        model.optimize()
        
        # Check solution status
        if model.status == GRB.OPTIMAL:
            # Extract dual information
            dual_values = {}
            
            # Get dual values for capacity constraints
            for t in range(self.params.total_periods):
                capacity_constr = model.getConstrByName(f"capacity[{t}]")
                if capacity_constr is not None:
                    # Store the dual value, particularly for linking periods
                    if t in self.params.linking_periods:
                        dual_values[f"capacity_dual_{t}"] = capacity_constr.Pi
            
            # Extract primal solution
            primal_solution = {
                'setup': [setup[t].x > 0.5 for t in range(self.params.total_periods)],
                'production': [production[t].x for t in range(self.params.total_periods)],
                'inventory': [inventory[t].x for t in range(self.params.total_periods)]
            }
            
            return model.objVal, dual_values, [], primal_solution
            
        elif model.status == GRB.INFEASIBLE:
            # Extract extreme ray information
            model.computeIIS()
            dual_rays = []
            
            # Create a simplified ray based on the IIS
            ray_coeffs = [0.0] * len(self.params.linking_periods)
            ray_const = 0.0
            
            # Build the ray from the IIS constraints
            for c in model.getConstrs():
                if c.IISConstr:
                    # For capacity constraints
                    if "capacity" in c.ConstrName:
                        period = int(c.ConstrName.split('[')[1].split(']')[0])
                        if period in self.params.linking_periods:
                            idx = sorted(self.params.linking_periods).index(period)
                            if fixed_setups.get(period, False):
                                ray_coeffs[idx] -= self.params.capacity  # Negative coefficient
                            else:
                                ray_coeffs[idx] += self.params.capacity  # Positive coefficient
            
            # Add a constant based on demand
            total_demand = sum(scenario.demands)
            ray_const = total_demand * 0.1  # Small portion of total demand
            
            # Add this ray if it has non-zero coefficients
            if any(abs(coef) > 1e-6 for coef in ray_coeffs):
                dual_rays.append((ray_coeffs, ray_const))
            
            return float('inf'), {}, dual_rays, None
            
        else:
            # Other status (unexpected)
            return float('inf'), {}, [], None
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6) -> Tuple[float, float, TimingStats]:
        """Execute Benders decomposition until gap < tolerance or max iters."""
        start = time.time()
        master_time = 0.0
        sub_time = 0.0
        ml_time = 0.0
        lap_time = 0.0

        # Reset ML statistics
        self.ml_attempts = 0
        self.ml_successes = 0
        self.ml_prediction_time = 0.0

        self.build_master_problem()
        best_lb, best_ub = -np.inf, np.inf
        best_sol = None
        iteration = 0
        stagnant = 0
        last_lb = -np.inf

        while iteration < max_iterations and (best_ub - best_lb) > tolerance:
            try:
                # Add/update trust region constraint
                if self.use_trust_region:
                    self.update_trust_region_constraint(iteration)

                # Solve master
                t0 = time.time()
                self.master_model.optimize()
                master_time += time.time() - t0

                if self.master_model.status != GRB.OPTIMAL:
                    if iteration == 0:
                        raise ValueError("Master infeasible at iteration 0")
                    break

                lb = self.master_model.objVal
                if lb > last_lb + 1e-6:
                    stagnant = 0
                    last_lb = lb
                else:
                    stagnant += 1

                # Extract first-stage
                setups, prods = self.extract_master_solution()

                # Solve subproblems
                t1 = time.time()
                total_cost = 0.0
                rec_solution = {'setup': [], 'production': [], 'inventory': []}
                all_feasible = True
                cuts_added = False

                for idx, prob in enumerate(self.scenario_problems):
                    theta_var = self.master_model.getVarByName(f"theta[{idx}]")
                    if theta_var is None:
                        continue
                    
                    # Get the theta value from the master
                    theta_val = theta_var.x

                    # Get the subproblem solution (using ML if enabled)
                    if self.use_ml:
                        self.ml_attempts += 1
                        t_ml_start = time.time()
                        ml_cost, rec = prob.solve_recursive(setups)
                        ml_end_time = time.time()
                        ml_elapsed = ml_end_time - t_ml_start
                        ml_time += ml_elapsed
                        
                        # Update ML statistics
                        if isinstance(prob, MLScenarioSubproblem):
                            self.ml_prediction_time += prob.ml_prediction_time
                            # Check ML success rate
                            if prob.ml_success > 0:
                                self.ml_successes += 1
                        elif isinstance(prob, HybridScenarioSubproblem):
                            stats = prob.get_hybrid_stats()
                            self.ml_prediction_time += stats['ml_prediction_time']
                            # Check ML success rate
                            if stats['success'] > 0:
                                self.ml_successes += 1
                        
                        if ml_cost < float('inf'):
                            # ML found a feasible solution
                            c = ml_cost
                            rec_solution['setup'].append(rec['setup'])
                            rec_solution['production'].append(rec['production'])
                            rec_solution['inventory'].append(rec['inventory'])
                            
                            # Check if optimality cut is needed
                            if theta_val < c - tolerance:
                                # Add optimality cut based on whether valid cuts are enabled
                                if self.use_valid_cuts:
                                    # Always solve LP relaxation for valid cuts
                                    lp_obj, dual_values, _, _ = self.solve_lp_relaxation(idx, setups)
                                    cuts_added = self.add_valid_optimality_cut(
                                        idx, dual_values, c) or cuts_added
                                else:
                                    # Use heuristic cuts only when valid cuts are disabled
                                    cuts_added = self.add_heuristic_optimality_cut(
                                        idx, c, setups) or cuts_added
                                
                                # Add Laporte cuts if enabled
                                if self.use_laporte_cuts and self.laporte_cuts_generator is not None:
                                    t2 = time.time()
                                    try:
                                        laporte_constr = self.laporte_cuts_generator.add_laporte_cut(
                                            self.master_model, idx, setups, c, self.num_laporte_cuts
                                        )
                                        if laporte_constr:
                                            # Set the Lazy attribute on the Laporte cut
                                            if hasattr(laporte_constr, "Lazy"):
                                                laporte_constr.Lazy = 1
                                            self.laporte_cuts.append(laporte_constr)
                                            self.num_laporte_cuts += 1
                                            cuts_added = True
                                    except Exception as e:
                                        print(f"Laporte cut generation error: {e}")
                                    lap_time += time.time() - t2
                        else:
                            # ML couldn't find a feasible solution
                            # Should not happen with top-K approach in normal operation
                            all_feasible = False
                            
                            # Try to add a feasibility cut based on LP relaxation
                            lp_obj, _, dual_rays, _ = self.solve_lp_relaxation(idx, setups)
                            cuts_added = self.add_feasibility_cut(idx, dual_rays) or cuts_added
                            break
                    else:
                        # No ML, use regular solver
                        c, rec = prob.solve_recursive(setups)
                        
                        if c < float('inf'):
                            # MIP found a feasible solution
                            rec_solution['setup'].append(rec['setup'])
                            rec_solution['production'].append(rec['production'])
                            rec_solution['inventory'].append(rec['inventory'])
                            
                            if theta_val < c - tolerance:
                                # Add optimality cut based on whether valid cuts are enabled
                                if self.use_valid_cuts:
                                    # Always solve LP relaxation for valid cuts
                                    lp_obj, dual_values, _, _ = self.solve_lp_relaxation(idx, setups)
                                    cuts_added = self.add_valid_optimality_cut(
                                        idx, dual_values, c) or cuts_added
                                else:
                                    # Use heuristic cuts only when valid cuts are disabled
                                    cuts_added = self.add_heuristic_optimality_cut(
                                        idx, c, setups) or cuts_added
                                
                                # Add Laporte cuts if enabled
                                if self.use_laporte_cuts and self.laporte_cuts_generator is not None:
                                    t2 = time.time()
                                    try:
                                        laporte_constr = self.laporte_cuts_generator.add_laporte_cut(
                                            self.master_model, idx, setups, c, self.num_laporte_cuts
                                        )
                                        if laporte_constr:
                                            # Set the Lazy attribute on the Laporte cut
                                            if hasattr(laporte_constr, "Lazy"):
                                                laporte_constr.Lazy = 1
                                            self.laporte_cuts.append(laporte_constr)
                                            self.num_laporte_cuts += 1
                                            cuts_added = True
                                    except Exception as e:
                                        print(f"Laporte cut generation error: {e}")
                                    lap_time += time.time() - t2
                        else:
                            # Subproblem is infeasible
                            all_feasible = False
                            lp_obj, _, dual_rays, _ = self.solve_lp_relaxation(idx, setups)
                            cuts_added = self.add_feasibility_cut(idx, dual_rays) or cuts_added
                            break
                    
                    # Add scenario contribution to total cost
                    total_cost += (self.params.scenarios[idx].probability * c)

                sub_time += time.time() - t1

                # Update bounds
                if all_feasible and total_cost < best_ub:
                    best_ub = total_cost
                    best_sol = rec_solution
                best_lb = max(best_lb, lb)

                # Make sure model is updated to reflect new constraints
                self.master_model.update()

                iteration += 1
                if iteration % 5 == 0 or iteration == 1:
                    gap = ((best_ub - best_lb) / best_ub * 100) if best_ub < np.inf else np.inf
                    print(f"Iter {iteration}: LB={best_lb:.2f}, UB={best_ub:.2f}, Gap={gap:.2f}%")
                    
                    # Print ML statistics if using ML
                    if self.use_ml and self.ml_attempts > 0:
                        ml_success_rate = self.ml_successes / self.ml_attempts * 100
                        print(f"  ML success rate: {ml_success_rate:.1f}% ({self.ml_successes}/{self.ml_attempts})")
                
                if stagnant > 10 or not cuts_added:
                    print("No improvement or no new cuts; terminating loop.")
                    break

            except Exception as e:
                print(f"Error at iter {iteration}: {e}")
                iteration += 1
                continue

        total = time.time() - start
        if best_sol is None:
            raise ValueError("Benders did not find any feasible solution.")

        # Calculate ML success rate
        if self.ml_attempts > 0:
            self.ml_success_rate = self.ml_successes / self.ml_attempts * 100
        else:
            self.ml_success_rate = 0.0

        # finalize stats and solution
        self.timing_stats = TimingStats(
            total_solve_time=total,
            master_time=master_time,
            subproblem_time=sub_time,
            ml_prediction_time=ml_time,
            laporte_cuts_time=lap_time,
            num_iterations=iteration
        )
        self.best_solution = Solution(
            setup=best_sol['setup'],
            production=best_sol['production'],
            inventory=best_sol['inventory'],
            objective=best_ub
        )

        # Print summary statistics
        print(f"\nBenders finished in {total:.2f} seconds")
        print(f"  Iterations: {iteration}")
        print(f"  Lower Bound: {best_lb:.2f}")
        print(f"  Upper Bound: {best_ub:.2f}")
        print(f"  Gap: {((best_ub - best_lb) / best_ub * 100):.2f}%")
        print(f"  Feasibility cuts: {self.num_feasibility_cuts}")
        
        if self.use_valid_cuts:
            print(f"  Valid optimality cuts: {self.num_valid_optimality_cuts}")
        else:
            print(f"  Heuristic optimality cuts: {self.num_heuristic_optimality_cuts}")
            
        if self.use_trust_region:
            print(f"  Trust region cuts: {self.num_trust_region_cuts}")
            
        if self.use_laporte_cuts:
            print(f"  Laporte cuts: {self.num_laporte_cuts}")
            
        if self.use_ml:
            print(f"\nML Statistics:")
            print(f"  ML success rate: {self.ml_success_rate:.1f}% ({self.ml_successes}/{self.ml_attempts})")
            print(f"  ML prediction time: {self.ml_prediction_time:.2f} seconds ({(self.ml_prediction_time/total*100):.1f}% of total)")
            print(f"  Total ML-related time: {ml_time:.2f} seconds ({(ml_time/total*100):.1f}% of total)")

        return best_lb, best_ub, self.timing_stats
        
    def get_ml_statistics(self) -> Dict:
        """
        Get statistics about ML prediction performance.
        
        Returns:
            Dictionary with ML statistics
        """
        return {
            'attempts': self.ml_attempts,
            'successes': self.ml_successes,
            'success_rate': self.ml_success_rate,
            'prediction_time': self.ml_prediction_time
        }
        
    def get_cut_statistics(self) -> Dict:
        """
        Get statistics about cuts generated during Benders decomposition.
        
        Returns:
            Dictionary with cut statistics
        """
        return {
            'feasibility_cuts': self.num_feasibility_cuts,
            'valid_optimality_cuts': self.num_valid_optimality_cuts,
            'heuristic_optimality_cuts': self.num_heuristic_optimality_cuts,
            'trust_region_cuts': self.num_trust_region_cuts,
            'laporte_cuts': self.num_laporte_cuts
        }