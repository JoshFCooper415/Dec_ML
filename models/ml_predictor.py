import os
import numpy as np
import torch
from .neural_network import ProductionPlanningNN

class MLSubproblemPredictor:
    """Class to handle ML model predictions for subproblems."""
    
    def __init__(self, model_dir='.'):
        """Initialize the ML predictor.
        
        Args:
            model_dir: Directory containing trained model files
        """
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models(model_dir)
        
    def load_models(self, model_dir):
        """Load all available ML models for different block sizes.
        
        Args:
            model_dir: Directory containing trained model files
        
        Raises:
            ValueError: If no model files are found
        """
        model_files = [f for f in os.listdir(model_dir) if f.startswith("production_planning_model_") and f.endswith(".pt")]
        
        if not model_files:
            raise ValueError("No model files found in the specified directory")
        
        for model_file in model_files:
            # Extract block size from filename
            block_size = int(model_file.split("block")[1].split(".")[0])
            
            # Load model checkpoint
            checkpoint = torch.load(os.path.join(model_dir, model_file), map_location=self.device)
            
            # Initialize model architecture
            model = ProductionPlanningNN(
                input_dim=checkpoint['input_dim'],
                num_periods=checkpoint['num_periods'],
                hidden_dim=checkpoint['hidden_dim']
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            self.models[block_size] = model
            print(f"Loaded model for block size {block_size} from {model_file}")
    
    def predict_subproblem_solution(self, features, block_size):
        """Make predictions for a subproblem using the appropriate model.
        
        Args:
            features: Feature vector for the subproblem
            block_size: Size of the time block
            
        Returns:
            Tuple of (setup, production, inventory) predictions
            
        Raises:
            ValueError: If no model is available for the specified block size
        """
        if block_size not in self.models:
            raise ValueError(f"No model available for block size {block_size}")
        
        model = self.models[block_size]
        
        # Convert features to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            setup_pred, production_pred, inventory_pred = model(features_tensor)
        
        # Convert to numpy and apply appropriate transformations
        setup = (setup_pred.squeeze().cpu().numpy() > 0.5)  # Convert to binary
        production = production_pred.squeeze().cpu().numpy()
        inventory = inventory_pred.squeeze().cpu().numpy()
        
        return setup, production, inventory
    
    def prepare_features(self, fixed_setups, demands, initial_inventory, params, block_size, start_period):
        """Prepare feature vector for model input.
        
        Args:
            fixed_setups: Dictionary of setup decisions fixed by the master problem
            demands: List of demands for all periods
            initial_inventory: Initial inventory level for the time block
            params: Problem parameters
            block_size: Size of the time block
            start_period: Starting period of the time block
            
        Returns:
            Feature vector for the model
        """
        # Create fixed setup array
        fixed_setup_array = np.zeros(block_size)
        for t in range(start_period, start_period + block_size):
            if t in fixed_setups:
                rel_t = t - start_period
                fixed_setup_array[rel_t] = 1 if fixed_setups[t] else 0
        
        # Get demands for this block
        demands_array = np.array(demands[start_period:start_period + block_size])
        
        # Problem parameters
        params_array = np.array([
            params.capacity,
            params.fixed_cost,
            params.holding_cost,
            params.production_cost,
            initial_inventory
        ])
        
        # Concatenate all features
        features = np.concatenate([
            fixed_setup_array,
            params_array,
            demands_array,
            np.array([initial_inventory])
        ])
        
        return features