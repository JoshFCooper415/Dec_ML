from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, NamedTuple

@dataclass
class ScenarioData:
    """Data class for storing scenario information."""
    demands: List[float]
    probability: float

@dataclass
class ProblemParameters:
    """Data class for storing problem parameters."""
    total_periods: int
    periods_per_block: int
    capacity: float
    fixed_cost: float
    holding_cost: float
    production_cost: float
    scenarios: List[ScenarioData]
    linking_periods: Set[int]

@dataclass
class TimingStats:
    """Data class for tracking timing information."""
    total_solve_time: float
    subproblem_time: float
    master_time: float
    num_iterations: int
    ml_prediction_time: float = 0.0  # New field for ML prediction time

class Solution(NamedTuple):
    """NamedTuple for storing solution data."""
    setup: List[List[bool]]
    production: List[List[float]]
    inventory: List[List[float]]
    objective: float