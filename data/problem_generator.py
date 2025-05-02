import random
from data.data_structures import ScenarioData, ProblemParameters

def create_test_problem(
    num_periods: int = 90, 
    num_scenarios: int = 3, 
    periods_per_block: int = 30,
    capacity_to_demand_ratio: int = 5,  # C parameter (values used in paper: 3, 5, 8)
    setup_to_holding_ratio: int = 1000,  # F parameter (values used in paper: 1000, 10000)
    seed: int = 42
) -> ProblemParameters:
    """Create a test problem instance following the paper's approach.
    
    Args:
        num_periods: Number of time periods (paper used T=90)
        num_scenarios: Number of scenarios to generate
        periods_per_block: Number of periods per planning block
        capacity_to_demand_ratio: C parameter - ratio of capacity to demand (paper used 3, 5, 8)
        setup_to_holding_ratio: F parameter - ratio of setup to holding cost (paper used 1000, 10000)
        seed: Random seed for reproducibility
        
    Returns:
        ProblemParameters object with the created test problem
    """
    random.seed(seed)
    
    scenarios = []
    holding_cost = 1  # Paper sets h_t = 1
    
    # Generate random demands following the paper's method
    # Paper uses d_t in range [1, 600]
    for _ in range(num_scenarios):
        demands = [
            random.randint(1, 600)
            for _ in range(num_periods)
        ]
        scenarios.append(ScenarioData(
            demands=demands,
            probability=1.0/num_scenarios
        ))
    
    # Generate production costs following the paper's method
    # Paper uses p_t in range [1, 5]
    production_cost = random.randint(1, 5)
    
    # Generate capacity following the paper's method
    # Paper uses c_t in range [0.7*c, 1.1*c]
    # Calculate the average demand across all scenarios and periods
    total_demand = sum(sum(scenario.demands) for scenario in scenarios)
    base_capacity = capacity_to_demand_ratio * (total_demand / (num_scenarios * num_periods))
    capacity = random.uniform(0.7 * base_capacity, 1.1 * base_capacity)
    
    # Generate setup cost following the paper's method
    # Paper uses f_t in range [0.9*f, 1.1*f]
    setup_cost = random.uniform(0.9 * setup_to_holding_ratio, 
                              1.1 * setup_to_holding_ratio)
    
    # Define linking periods (every n periods based on periods_per_block)
    linking_periods = set(range(0, num_periods, periods_per_block))
    
    return ProblemParameters(
        total_periods=num_periods,
        periods_per_block=periods_per_block,
        capacity=capacity,
        fixed_cost=setup_cost,
        holding_cost=holding_cost,
        production_cost=production_cost,
        scenarios=scenarios,
        linking_periods=linking_periods
    )