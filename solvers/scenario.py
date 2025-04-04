from typing import Dict, Tuple, Set, List
import time

from data.data_structures import ScenarioData, ProblemParameters
from models.ml_predictor import MLSubproblemPredictor
from .timeblock import MLTimeBlockSubproblem

class MLScenarioSubproblem:
    """Scenario subproblem solver with ML enhancement."""
    
    def __init__(self, scenario_data: ScenarioData, params: ProblemParameters, 
                ml_predictor: MLSubproblemPredictor = None):
        """Initialize the scenario subproblem.
        
        Args:
            scenario_data: Scenario data for the subproblem
            params: Problem parameters
            ml_predictor: ML predictor for solution prediction
        """
        self.scenario_data = scenario_data
        self.params = params
        self.ml_predictor = ml_predictor
        self.time_blocks: List[MLTimeBlockSubproblem] = []
        self._create_time_blocks()
    
    def _create_time_blocks(self):
        """Create time block subproblems for this scenario."""
        num_blocks = (self.params.total_periods + self.params.periods_per_block - 1) 
        num_blocks //= self.params.periods_per_block
        
        for i in range(num_blocks):
            start = i * self.params.periods_per_block
            periods = min(self.params.periods_per_block, 
                         self.params.total_periods - start)
            block = MLTimeBlockSubproblem(
                start, periods, self.scenario_data, self.params, self.ml_predictor
            )
            self.time_blocks.append(block)
    
    def solve_recursive(self, fixed_setups: Dict[int, bool]) -> Tuple[float, Dict]:
        """Solve scenario with fixed setups from master problem using ML where possible.
        
        Args:
            fixed_setups: Dictionary of setup decisions fixed by the master problem
            
        Returns:
            Tuple of (total_cost, solution_data)
        """
        inventory_values = {-1: 0}
        total_cost = 0
        solution_data = {
            'setup': [False] * self.params.total_periods,
            'production': [0.0] * self.params.total_periods,
            'inventory': [0.0] * (self.params.total_periods + 1)
        }
        
        for i, block in enumerate(self.time_blocks):
            initial_inv = inventory_values[i-1]
            
            # Try ML prediction first
            if self.ml_predictor is not None:
                ml_start = time.time()
                ml_cost, ml_feasible = block.predict_solution(initial_inv, fixed_setups)
                ml_time = time.time() - ml_start
                
                if ml_feasible:
                    # ML solution is feasible, use it
                    block_cost = ml_cost
                    is_feasible = True
                    block_sol = block.get_solution()
                else:
                    # ML failed, fall back to Gurobi
                    block.build_model(initial_inv, fixed_setups)
                    block_cost, is_feasible = block.solve()
                    block_sol = block.get_solution()
            else:
                # No ML predictor, use Gurobi directly
                block.build_model(initial_inv, fixed_setups)
                block_cost, is_feasible = block.solve()
                block_sol = block.get_solution()
            
            if not is_feasible:
                return float('inf'), {}
            
            # Extract solution
            start_idx = block.start_period
            
            for t in range(block.num_periods):
                period = start_idx + t
                solution_data['setup'][period] = block_sol['setup'][t]
                solution_data['production'][period] = block_sol['production'][t]
                solution_data['inventory'][period] = block_sol['inventory'][t]
            
            if i == len(self.time_blocks) - 1:
                solution_data['inventory'][-1] = block_sol['inventory'][-1]
            
            inventory_values[i] = block_sol['inventory'][-1]
            total_cost += block_cost
        
        return total_cost, solution_data
    
    def get_critical_setups(self) -> Set[int]:
        """Collect critical setups from all blocks.
        
        Returns:
            Set of critical setup periods
        """
        critical = set()
        for block in self.time_blocks:
            critical.update(block.get_critical_setups())
        return critical