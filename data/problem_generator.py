import random
from data.data_structures import ScenarioData, ProblemParameters

def create_test_problem(num_periods: int = 12, num_scenarios: int = 3, seed: int = 42) -> ProblemParameters:
    """Create a test problem instance.
    
    Args:
        num_periods: Number of time periods
        num_scenarios: Number of scenarios to generate
        seed: Random seed for reproducibility
        
    Returns:
        ProblemParameters object with the created test problem
    """
    random.seed(seed)
    
    scenarios = []
    
    # Generate random demand scenarios
    base_demand = 100
    for _ in range(num_scenarios):
        demands = [
            base_demand * (1 + 0.3 * random.uniform(-1, 1))
            for _ in range(num_periods)
        ]
        scenarios.append(ScenarioData(
            demands=demands,
            probability=1.0/num_scenarios
        ))
    
    # Define linking periods (every third period)
    linking_periods = set(range(0, num_periods, 3))
    
    return ProblemParameters(
        total_periods=num_periods,
        periods_per_block=3,
        capacity=300,
        fixed_cost=1000,
        holding_cost=5,
        production_cost=10,
        scenarios=scenarios,
        linking_periods=linking_periods
    )