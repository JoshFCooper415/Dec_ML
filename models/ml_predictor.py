from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import os
import time

from models.neural_network import ProductionPlanningNN
from data.data_structures import ProblemParameters, ScenarioData


class MLSubproblemPredictor:
    """ML predictor for production planning subproblems with top-K solutions capability"""

    def __init__(self, model_path='models', k=3):
        """
        Initialize the ML predictor with top-K capability.
        
        Args:
            model_path: Path to model files
            k: Number of top solutions to generate (default: 3)
        """
        self.model = None
        self.is_loaded = False
        self.input_dim = None
        self.num_periods = None
        self.hidden_dim = None
        self.k = k  # Store the number of solutions to generate
        self.prediction_time = 0.0  # Track total prediction time

        # Load available model files
        self.available_models = []
        if isinstance(model_path, str) and os.path.isdir(model_path):
            self.available_models = [
                os.path.join(model_path, f)
                for f in os.listdir(model_path)
                if f.endswith((".pth", ".pt"))
            ]

        # Attempt to load models
        for path in self.available_models:
            try:
                checkpoint = torch.load(path, map_location='cpu',weights_only=False)

                if 'model_state_dict' in checkpoint:
                    self._create_model_from_checkpoint(checkpoint)
                    print(f"Successfully loaded model from {path}")
                    self.is_loaded = True
                    break

            except Exception as e:
                print(f"Error loading {path}: {str(e)}")

        if not self.is_loaded:
            print("Warning: No valid model file found in specified path.")
            self._create_default_mlp()

    def _create_model_from_checkpoint(self, checkpoint):
        """Helper to create model from checkpoint"""
        self.input_dim = checkpoint['input_dim']
        self.num_periods = checkpoint['num_periods']
        self.hidden_dim = checkpoint.get('hidden_dim', 128)

        # Basic validation
        expected_input_dim = 2 * self.num_periods + 5
        if self.input_dim != expected_input_dim:
            print(f"Warning: Loaded model input_dim ({self.input_dim}) does not match expected ({expected_input_dim}) for num_periods={self.num_periods}.")

        model_type = checkpoint.get('model_type', 'standard')
        if model_type == 'transformer':
             from models.transformer_neural_network import TransformerProductionPlanningNN
             num_heads = checkpoint.get('num_heads', 4)
             num_layers = checkpoint.get('num_layers', 2)
             self.model = TransformerProductionPlanningNN(
                 input_dim=self.input_dim,
                 num_periods=self.num_periods,
                 hidden_dim=self.hidden_dim,
                 num_heads=num_heads,
                 num_layers=num_layers
             )
             print("Loaded Transformer model.")
        else:
             self.model = ProductionPlanningNN(
                 input_dim=self.input_dim,
                 num_periods=self.num_periods,
                 hidden_dim=self.hidden_dim
             )
             print("Loaded Standard NN model.")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def _create_default_mlp(self):
        """Create MLP with safe defaults if no model file is loaded"""
        self.num_periods = 30
        self.input_dim = 2 * self.num_periods + 5
        self.hidden_dim = 128
        print(f"Warning: No model file found. Creating default MLP with block size {self.num_periods} and input dim {self.input_dim}.")

        self.model = ProductionPlanningNN(
            input_dim=self.input_dim,
            num_periods=self.num_periods,
            hidden_dim=self.hidden_dim
        )
        self.model.eval()
        self.is_loaded = True

    def prepare_features(self, fixed_setups: Dict[int, bool], scenario: ScenarioData, 
                        params: ProblemParameters, start_period: int, initial_inventory: float):
        """Prepare input features for the ML model, matching training data structure."""
        if not self.is_loaded or self.model is None:
             print("Error: Attempting to prepare features, but ML model is not loaded.")
             return None

        block_size = self.num_periods

        # 1. Setup features (binary) - shape (block_size,)
        setup_vector = np.zeros(block_size, dtype=np.float32)
        for t in range(block_size):
            period_in_full_horizon = start_period + t
            if period_in_full_horizon in params.linking_periods:
                 setup_vector[t] = 1.0 if fixed_setups.get(period_in_full_horizon, False) else 0.0

        # 2. Problem parameters - shape (4,)
        params_vector = np.array([
            params.capacity,
            params.fixed_cost,
            params.holding_cost,
            params.production_cost
        ], dtype=np.float32)

        # 3. Demand values - shape (block_size,)
        demands = np.zeros(block_size, dtype=np.float32)
        end_period_in_block = min(start_period + block_size, len(scenario.demands))
        actual_demands_in_block = scenario.demands[start_period : end_period_in_block]
        demands[:len(actual_demands_in_block)] = actual_demands_in_block

        # 4. Initial inventory - shape (1,)
        init_inv_vector = np.array([initial_inventory], dtype=np.float32)

        # Combine all features
        feature_vector = np.concatenate([
            setup_vector,
            params_vector,
            demands,
            init_inv_vector
        ])

        # Verify final dimension
        if len(feature_vector) != self.input_dim:
             print(f"CRITICAL Error: Prepared feature vector length ({len(feature_vector)}) does not match loaded model input dimension ({self.input_dim}).")
             print(f"Debug Info: block_size={block_size}, setup={len(setup_vector)}, params={len(params_vector)}, demands={len(demands)}, init_inv={len(init_inv_vector)}")
             return None

        return torch.FloatTensor(feature_vector).unsqueeze(0)  # Add batch dimension

    def generate_alternative_setups(self, setup_probs: np.ndarray, num_alternatives: int) -> List[np.ndarray]:
        """
        Generate alternative setup decisions based on probability scores.
        
        Args:
            setup_probs: NumPy array of setup probabilities from the model
            num_alternatives: Number of alternatives to generate
            
        Returns:
            List of NumPy arrays with alternative binary setup decisions
        """
        # Start with the deterministic (threshold 0.5) solution
        setup_binary_base = (setup_probs > 0.5).astype(float)
        alternatives = [setup_binary_base]
        
        # Find periods with most uncertain setup decisions (probabilities closest to 0.5)
        uncertainty = np.abs(setup_probs - 0.5)
        uncertain_indices = np.argsort(uncertainty)
        
        # Generate alternatives by flipping decisions at the most uncertain points
        for i in range(1, num_alternatives):
            if i <= len(uncertain_indices):
                # Create a copy of the base setup decision
                alt_setup = setup_binary_base.copy()
                
                # Flip the i-th most uncertain setup decision
                flip_idx = uncertain_indices[i-1]
                alt_setup[flip_idx] = 1.0 - alt_setup[flip_idx]
                
                # Add this alternative to the list
                alternatives.append(alt_setup)
                
        return alternatives
    
    def generate_diverse_alternatives(self, setup_probs: np.ndarray, num_alternatives: int) -> List[np.ndarray]:
        """
        Generate diverse alternative setup decisions using stochastic sampling.
        
        Args:
            setup_probs: NumPy array of setup probabilities from the model
            num_alternatives: Number of alternatives to generate
            
        Returns:
            List of NumPy arrays with diverse binary setup decisions
        """
        # Start with the deterministic (threshold 0.5) solution
        setup_binary_base = (setup_probs > 0.5).astype(float)
        alternatives = [setup_binary_base]
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate additional alternatives through sampling from probabilities
        for i in range(1, num_alternatives):
            # Sample from Bernoulli distribution based on predicted probabilities
            random_sample = np.random.random(len(setup_probs))
            alt_setup = (random_sample < setup_probs).astype(float)
            
            # Make sure this sample is different from previous ones
            is_duplicate = False
            for prev_alt in alternatives:
                if np.array_equal(alt_setup, prev_alt):
                    is_duplicate = True
                    break
                    
            # If it's a duplicate, try to modify it until it's unique
            if is_duplicate:
                # Find indices where the probability is close to 0.5 (most uncertain)
                uncertainty = np.abs(setup_probs - 0.5)
                uncertain_indices = np.argsort(uncertainty)
                
                # Flip the most uncertain bit
                if len(uncertain_indices) > 0:
                    flip_idx = uncertain_indices[0]
                    alt_setup[flip_idx] = 1.0 - alt_setup[flip_idx]
            
            # Add to alternatives list
            alternatives.append(alt_setup)
            
        return alternatives

    def predict_top_k_solutions(self, features, num_periods):
        """
        Predict top-K alternative solutions for the subproblem.
        
        Args:
            features: Input features tensor
            num_periods: Number of periods in the current block
            
        Returns:
            List of K tuples (setup_binary, production_plan, inventory_plan)
            Each tuple is a candidate solution. Returns empty list on failure.
        """
        start_time = time.time()
        
        if not self.is_loaded or self.model is None:
             print("Error: Cannot predict, ML model is not loaded.")
             return []
        if features is None:
             print("Error: Cannot predict, features are None.")
             return []

        try:
            with torch.no_grad():
                # Get base prediction from model
                setup_pred, production_pred, inventory_pred = self.model(features)
                
                # Extract setup probabilities from sigmoid output (not binary yet)
                setup_probs = setup_pred.squeeze().numpy()
                
                # Convert production and inventory to numpy
                production_np = production_pred.squeeze().numpy()
                inventory_np = inventory_pred.squeeze().numpy()
                
                # Ensure outputs match the expected number of periods
                current_block_actual_periods = min(num_periods, self.num_periods)
                
                # Keep only predictions relevant to this block
                setup_probs = setup_probs[:current_block_actual_periods]
                production_np = production_np[:current_block_actual_periods]
                inventory_np = inventory_np[:current_block_actual_periods]
                
                # Generate alternative setup decisions with two different methods
                flip_based_alts = self.generate_alternative_setups(setup_probs, self.k)
                diverse_alts = self.generate_diverse_alternatives(setup_probs, self.k)
                
                # Combine both types of alternatives, eliminating duplicates
                all_setups = []
                for setup in flip_based_alts + diverse_alts:
                    # Check if this setup is already in our list
                    is_duplicate = False
                    for existing_setup in all_setups:
                        if np.array_equal(setup, existing_setup):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_setups.append(setup)
                
                # Create final solutions list with unique setups
                solutions = []
                for setup in all_setups[:self.k]:  # Limit to k solutions
                    solutions.append((setup, production_np, inventory_np))
                
                # Update prediction time
                self.prediction_time = time.time() - start_time
                
                # Return the solutions
                return solutions

        except Exception as e:
            print(f"Top-K prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    # Keep the original method for backward compatibility
    def predict_subproblem_solution(self, features, num_periods):
        """
        Original prediction method that returns only the best solution.
        
        Args:
            features: Input features tensor
            num_periods: Number of periods in the current block
            
        Returns:
            Tuple of (setup_binary, production_np, inventory_np) or (None, None, None) on failure
        """
        solutions = self.predict_top_k_solutions(features, num_periods)
        if solutions:
            return solutions[0]  # Return the first (highest probability) solution
        else:
            return None, None, None
    
    def set_k(self, new_k: int):
        """
        Change the number of alternative solutions to generate.
        
        Args:
            new_k: New number of solutions to generate
        """
        if new_k > 0:
            self.k = new_k
            print(f"Updated predictor to generate {self.k} alternative solutions")
        else:
            print("Error: k must be a positive integer")