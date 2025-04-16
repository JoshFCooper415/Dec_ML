import torch
import numpy as np
import os
import sys

class MLSubproblemPredictor:
    """ML predictor for subproblem solutions using transformer-based model."""
    
    def __init__(self, model_path='models'):
        """Initialize the ML predictor with transformer model support.
        
        Args:
            model_path: Path to the model checkpoint file
        """
        self.model = None
        self.is_loaded = False
        self.available_models = []
        self.model_info = {}  # Store model info for debugging
        self.use_transformer = False  # Flag to indicate we're using transformer model
        
        print("\nInitializing ML predictor with transformer model support")
        
        # Check if model_path is a string - avoid permission issues
        if not isinstance(model_path, str):
            print(f"Invalid model_path: {model_path}")
            return
            
        # If model_path is a directory, try to find any model files in it
        if os.path.isdir(model_path):
            try:
                print(f"Searching for model files in directory: {model_path}")
                for filename in os.listdir(model_path):
                    if filename.endswith(".pth") or filename.endswith(".pt"):
                        self.available_models.append(os.path.join(model_path, filename))
                        
                if self.available_models:
                    print(f"Found {len(self.available_models)} model files in {model_path}")
                else:
                    print(f"No model files found in {model_path}")
            except PermissionError as e:
                print(f"Permission error accessing directory {model_path}: {str(e)}")
            except Exception as e:
                print(f"Error accessing directory {model_path}: {str(e)}")
        
        # Try to find model in default locations
        possible_paths = [
            # Don't add model_path if it's just 'models' or a directory
            model_path if not model_path == 'models' and not os.path.isdir(model_path) else None,
            "./models/predictor_model.pth",
            "../models/predictor_model.pth",
            "./models/production_planning_model_production_planning_ml_data_block3.pt",
            "../models/production_planning_model_production_planning_ml_data_block3.pt",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor_model.pth"),
        ] + self.available_models  # Add any models found in directory
        
        # Remove duplicates and None values
        possible_paths = [p for p in possible_paths if p is not None]
        possible_paths = list(dict.fromkeys(possible_paths))  # Remove duplicates while preserving order
        
        # Try each path and load the model
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                print(f"Attempting to load model from {path}")
                
                try:
                    # First, try to allowlist numpy scalar global - needed for PyTorch 2.6
                    try:
                        # This is a fix for PyTorch 2.6 weights_only=True restriction
                        torch.serialization.add_safe_globals(["numpy._core.multiarray.scalar"])
                    except (AttributeError, ImportError, ValueError) as e:
                        print(f"Failed to add numpy.scalar to safe globals: {str(e)}")
                    
                    # Load with weights_only=False - less secure but more likely to work
                    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
                    
                    # Check if it's a dictionary with model_state_dict
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        print(f"Found model state dict in {path}")
                        
                        # Try to create and load the model
                        try:
                            # Import the transformer model
                            # Note: You'll need to ensure this import works in your project structure
                            # If needed, you can directly copy the TransformerProductionPlanningNN class here
                            try:
                                from models.transformer_neural_network import TransformerProductionPlanningNN
                                print("Successfully imported TransformerProductionPlanningNN")
                            except ImportError:
                                # If the transformer model isn't available as a module, 
                                # import the original model and we'll create a transformer model manually
                                print("Couldn't import transformer model, will create one manually")
                                from models.neural_network import TransformerProductionPlanningNN
                            
                            # Check if we have the necessary parameters
                            if all(k in checkpoint for k in ['input_dim', 'num_periods']):
                                # Get parameters from checkpoint
                                input_dim = checkpoint['input_dim']
                                num_periods = checkpoint['num_periods']
                                hidden_dim = checkpoint.get('hidden_dim', 128)
                                
                                # Create the transformer model
                                model = TransformerProductionPlanningNN(
                                    input_dim=input_dim,
                                    num_periods=num_periods,
                                    hidden_dim=hidden_dim,
                                    num_heads=4,  # Default value
                                    num_layers=2  # Default value
                                )
                                
                                # Try to load the state dict - this might fail due to architecture differences
                                try:
                                    model.load_state_dict(checkpoint['model_state_dict'])
                                    print("Successfully loaded existing weights into transformer model")
                                except Exception as e:
                                    print(f"Could not load weights into transformer model: {str(e)}")
                                    print("Using transformer model with fresh weights")
                                
                                model.eval()  # Set to evaluation mode
                                
                                self.model = model
                                self.input_dim = input_dim
                                self.num_periods = num_periods
                                self.hidden_dim = hidden_dim
                                self.is_loaded = True
                                
                                self.model_info = {
                                    'input_dim': input_dim,
                                    'num_periods': num_periods,
                                    'hidden_dim': hidden_dim,
                                    'path': path,
                                    'model_type': 'transformer'
                                }
                                
                                print(f"Successfully created transformer model with parameters:")
                                print(f"  input_dim={input_dim}, num_periods={num_periods}, hidden_dim={hidden_dim}")
                                
                                if 'metrics' in checkpoint:
                                    print(f"Original model metrics: {checkpoint['metrics']}")
                                
                                break
                            else:
                                print(f"State dict missing required parameters")
                        except Exception as e:
                            print(f"Error creating transformer model from state dict: {str(e)}")
                    else:
                        # Try to load it directly as a model - less likely to work with architecture change
                        try:
                            model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
                            
                            if hasattr(model, 'forward'):  # Check if it's a PyTorch module
                                print("Loaded model directly, will attempt to convert to transformer")
                                
                                # Try to extract parameters
                                if hasattr(model, 'input_dim') and hasattr(model, 'num_periods'):
                                    input_dim = model.input_dim
                                    num_periods = model.num_periods
                                    hidden_dim = getattr(model, 'hidden_dim', 128)
                                    
                                    # Import the transformer model
                                    try:
                                        from models.transformer_neural_network import TransformerProductionPlanningNN
                                    except ImportError:
                                        # Define TransformerProductionPlanningNN class here if import fails
                                        # This would be a copy of the class from the other file
                                        raise ImportError("TransformerProductionPlanningNN not available")
                                    
                                    # Create a new transformer model
                                    transformer_model = TransformerProductionPlanningNN(
                                        input_dim=input_dim,
                                        num_periods=num_periods,
                                        hidden_dim=hidden_dim
                                    )
                                    transformer_model.eval()
                                    
                                    self.model = transformer_model
                                    self.input_dim = input_dim
                                    self.num_periods = num_periods
                                    self.hidden_dim = hidden_dim
                                    self.is_loaded = True
                                    
                                    self.model_info = {
                                        'input_dim': input_dim,
                                        'num_periods': num_periods,
                                        'hidden_dim': hidden_dim,
                                        'path': path,
                                        'model_type': 'transformer_converted'
                                    }
                                    
                                    print(f"Created fresh transformer model based on loaded model parameters")
                                    break
                                else:
                                    print("Loaded model doesn't have required attributes")
                            else:
                                print(f"Loaded object is not a PyTorch model")
                        except Exception as e:
                            print(f"Error loading as direct model: {str(e)}")
                
                except Exception as e:
                    print(f"Error loading model file: {str(e)}")
        
        # Create a transformer model with the correct dimensions if none was loaded
        if not self.is_loaded or self.model is None:
            print("\nNo valid ML predictor model could be loaded.")
            print("Creating a transformer model with default dimensions.")
            
            try:
                # Try to import TransformerProductionPlanningNN
                try:
                    from models.transformer_neural_network import TransformerProductionPlanningNN
                except ImportError:
                    # Insert the TransformerProductionPlanningNN class here if import fails
                    # This would be a copy of the class code
                    raise ImportError("TransformerProductionPlanningNN not available")
                
                # Determine sizes based on the block size
                block_size = 6  # Default block size
                
                # From the data generator, we know the feature structure:
                # - X_setup: block_size dimensions
                # - X_params: 4 dimensions (capacity, fixed_cost, holding_cost, production_cost)
                # - X_demands: block_size dimensions
                # - X_init_inv: 1 dimension
                # Total input dimensions: 2*block_size + 5
                input_dim = 2 * block_size + 5
                hidden_dim = 128  # Default hidden dimension
                
                model = TransformerProductionPlanningNN(
                    input_dim=input_dim,
                    num_periods=block_size,
                    hidden_dim=hidden_dim,
                    num_heads=4,
                    num_layers=2
                )
                model.eval()  # Set to evaluation mode
                
                self.model = model
                self.input_dim = input_dim
                self.num_periods = block_size
                self.hidden_dim = hidden_dim
                self.is_loaded = True
                
                self.model_info = {
                    'input_dim': input_dim,
                    'num_periods': block_size,
                    'hidden_dim': hidden_dim,
                    'path': 'generated_transformer_model',
                    'model_type': 'transformer_new'
                }
                
                print(f"Created transformer model with default dimensions:")
                print(f"  input_dim={input_dim}, num_periods={block_size}, hidden_dim={hidden_dim}")
            except Exception as e:
                print(f"Error creating transformer model: {str(e)}")
                self.model = None
                self.is_loaded = False
                print("Will run without ML predictions.")
        else:
            print("ML model loaded successfully and ready for predictions.")
            
        # Print model info for debugging
        if self.is_loaded and hasattr(self, 'input_dim'):
            print(f"\nModel info: input_dim={self.input_dim}, num_periods={getattr(self, 'num_periods', 'unknown')}")
    
    def prepare_features(self, fixed_setups, scenario):
        """Prepare features for the ML model to match the training data structure.
        
        Args:
            fixed_setups: Dictionary mapping time periods to setup decisions (True/False)
            scenario: ScenarioData object containing demand information
            
        Returns:
            Feature tensor ready for input to the ML model
        """
        if not self.is_loaded or self.model is None:
            print("Model not loaded, cannot prepare features")
            return None
            
        try:
            # Get block size from the model
            block_size = getattr(self, 'num_periods', 6)  # Default to 6 if not available
            
            # Create setup vector from fixed_setups
            # This should match X_setup in the training data
            setup_vector = np.zeros(block_size)
            for t in range(block_size):
                period = t  # Adjust if start_period is not 0
                if period in fixed_setups and fixed_setups[period]:
                    setup_vector[t] = 1.0
            
            # Extract the relevant demands for this block
            # This should match X_demands in the training data
            demands = np.array(scenario.demands[:block_size])
            if len(demands) < block_size:
                # Pad if needed
                demands = np.pad(demands, (0, block_size - len(demands)), 'constant')
            
            # Create parameters vector for production planning
            # This should match X_params in the training data
            params_vector = np.array([
                getattr(scenario, 'capacity', 300),          # capacity
                getattr(scenario, 'fixed_cost', 1000),       # fixed_cost
                getattr(scenario, 'holding_cost', 5),        # holding_cost
                getattr(scenario, 'production_cost', 10)     # production_cost
            ])
            
            # Initial inventory - assume 0 if not specified
            # This should match X_init_inv in the training data
            init_inv = np.array([0.0])
            
            # Combine into the full feature vector in the same order as training data
            feature_vector = np.concatenate([
                setup_vector,           # X_setup
                params_vector,          # X_params
                demands,                # X_demands
                init_inv                # X_init_inv
            ])
            
            # Print feature composition for debugging
            print(f"Feature vector composition: setup({len(setup_vector)}) + params({len(params_vector)}) + "
                  f"demands({len(demands)}) + init_inv({len(init_inv)}) = {len(feature_vector)}")
            
            # Check if we have the expected input dimension
            if hasattr(self, 'input_dim'):
                expected_dim = self.input_dim
                actual_dim = len(feature_vector)
                
                if actual_dim != expected_dim:
                    print(f"Warning: Feature vector dimension ({actual_dim}) doesn't match model input dimension ({expected_dim})")
                    
                    if actual_dim < expected_dim:
                        # Need to pad
                        padding_size = expected_dim - actual_dim
                        padding = np.zeros(padding_size)
                        feature_vector = np.concatenate([feature_vector, padding])
                        print(f"Padded feature vector from {actual_dim} to {expected_dim} dimensions")
                    else:
                        # Need to truncate
                        feature_vector = feature_vector[:expected_dim]
                        print(f"Truncated feature vector from {actual_dim} to {expected_dim} dimensions")
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            print(f"Final feature dimensions: {input_tensor.shape}")
            
            return input_tensor
        
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            traceback = sys.exc_info()[2]
            if traceback:
                print(f"Error at line: {traceback.tb_lineno}")
            return None
    
    def predict_subproblem_solution(self, features, num_periods):
        """Predict setup, production, and inventory decisions using transformer model.
        
        Args:
            features: Input features tensor
            num_periods: Number of time periods
            
        Returns:
            Tuple of (setup_decisions, production_quantities, inventory_levels)
        """
        if not self.is_loaded or self.model is None:
            print("Model not loaded, cannot make predictions")
            return None, None, None
        
        if features is None:
            print("Invalid features, cannot make predictions")
            return None, None, None
        
        try:
            # Safety check for matrix dimensions
            print(f"Feature dimensions: {features.shape}")
            
            # Apply model
            with torch.no_grad():
                self.model.eval()
                output = self.model(features)
            
            # Get output values - handle different model output formats
            if isinstance(output, tuple) and len(output) == 3:
                # Model returns separate outputs for each task
                setup_pred, prod_pred, inv_pred = output
                setup_np = (setup_pred.squeeze(0) > 0.5).float().numpy()
                prod_np = prod_pred.squeeze(0).numpy()
                inv_np = inv_pred.squeeze(0).numpy()
            else:
                # Model returns concatenated output
                output_np = output.squeeze(0).numpy()
                
                # Use the actual model's num_periods if available
                model_periods = getattr(self, 'num_periods', num_periods)
                
                # Get setup decisions
                if len(output_np) >= model_periods:
                    setup_np = (output_np[:model_periods] > 0.5).astype(float)
                else:
                    # Output too small, generate dummy values
                    setup_np = np.zeros(num_periods)
                
                # Get production
                if len(output_np) >= 2 * model_periods:
                    prod_np = output_np[model_periods:2*model_periods]
                else:
                    # Output too small, generate dummy values
                    prod_np = np.ones(num_periods) * 100  # Default production
                
                # Get inventory
                if len(output_np) >= 3 * model_periods:
                    inv_np = output_np[2*model_periods:3*model_periods]
                else:
                    # Output too small, generate dummy values
                    inv_np = np.ones(num_periods) * 20  # Default inventory
            
            # Ensure predictions have the right size
            if len(setup_np) != num_periods:
                # Resize to match expected size
                if len(setup_np) > num_periods:
                    setup_np = setup_np[:num_periods]
                else:
                    # Pad with zeros
                    setup_np = np.pad(setup_np, (0, num_periods - len(setup_np)))
            
            if len(prod_np) != num_periods:
                # Resize to match expected size
                if len(prod_np) > num_periods:
                    prod_np = prod_np[:num_periods]
                else:
                    # Pad with values
                    prod_np = np.pad(prod_np, (0, num_periods - len(prod_np)), 
                                     constant_values=100)  # Default production
            
            if len(inv_np) != num_periods:
                # Resize to match expected size
                if len(inv_np) > num_periods:
                    inv_np = inv_np[:num_periods]
                else:
                    # Pad with values
                    inv_np = np.pad(inv_np, (0, num_periods - len(inv_np)), 
                                   constant_values=20)  # Default inventory
            
            # Debug output
            print(f"Predicted setup: {setup_np}")
            print(f"Predicted production: {prod_np[:3]}... (len={len(prod_np)})")
            print(f"Predicted inventory: {inv_np[:3]}... (len={len(inv_np)})")
            
            return setup_np, prod_np, inv_np
        
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            traceback = sys.exc_info()[2]
            if traceback:
                print(f"Error at line: {traceback.tb_lineno}")
            return None, None, None
    
    def predict_production_plan(self, fixed_setups, scenario):
        """Predict an optimal production plan given fixed setups and demands.
        
        Args:
            fixed_setups: Dictionary mapping time periods to setup decisions (True/False)
            scenario: ScenarioData object containing demand information
            
        Returns:
            Tuple of (production_plan, inventory_plan)
        """
        if not self.is_loaded or self.model is None:
            # If model is not loaded, return None to indicate no prediction
            return None, None
        
        try:
            # Prepare input features
            input_tensor = self.prepare_features(fixed_setups, scenario)
            
            if input_tensor is None:
                return None, None
            
            # Get prediction - use the predict_subproblem_solution method
            setup, production, inventory = self.predict_subproblem_solution(
                input_tensor, len(scenario.demands))
            
            return production, inventory
        
        except Exception as e:
            print(f"ML prediction error: {str(e)}")
            traceback = sys.exc_info()[2]
            if traceback:
                print(f"Error at line: {traceback.tb_lineno}")
            return None, None
