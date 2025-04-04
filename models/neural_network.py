import torch
import torch.nn as nn

class ProductionPlanningNN(nn.Module):
    """Neural network model for production planning.
    
    This model uses a shared base network followed by three task-specific heads:
    1. Setup head: Binary classification for setup decisions
    2. Production head: Regression for production quantities
    3. Inventory head: Regression for inventory levels
    """
    
    def __init__(self, input_dim, num_periods, hidden_dim=128):
        """Initialize the neural network.
        
        Args:
            input_dim: Dimension of the input features
            num_periods: Number of time periods in the planning horizon
            hidden_dim: Dimension of hidden layers
        """
        super(ProductionPlanningNN, self).__init__()
        self.num_periods = num_periods
        
        # Common base layers
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Task-specific heads
        self.setup_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_periods),
            nn.Sigmoid()  # For binary classification
        )
        
        self.production_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_periods),
            nn.ReLU()  # Production is non-negative
        )
        
        self.inventory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_periods),
            nn.ReLU()  # Inventory is typically non-negative
        )
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (setup_output, production_output, inventory_output)
        """
        base_features = self.base(x)
        
        setup_output = self.setup_head(base_features)
        production_output = self.production_head(base_features)
        inventory_output = self.inventory_head(base_features)
        
        return setup_output, production_output, inventory_output