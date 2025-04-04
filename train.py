#!/usr/bin/env python
"""
Train neural network models for production planning.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Local imports
from models import ProductionPlanningNN

# Custom dataset for production planning data
class ProductionPlanningDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for production planning data."""
    
    def __init__(self, npz_file):
        """Initialize the dataset from a .npz file.
        
        Args:
            npz_file: Path to the .npz file containing training data
        """
        data = np.load(npz_file)
        self.X_setup = torch.FloatTensor(data['X_setup'])
        self.X_params = torch.FloatTensor(data['X_params'])
        self.X_demands = torch.FloatTensor(data['X_demands'])
        self.X_init_inv = torch.FloatTensor(data['X_init_inv']).unsqueeze(1)  # Add dimension for concatenation
        
        self.y_setup = torch.FloatTensor(data['y_setup'])
        self.y_production = torch.FloatTensor(data['y_production'])
        self.y_inventory = torch.FloatTensor(data['y_inventory'])
        
        self.problem_ids = data['problem_ids']
        
        # Get dimensions
        self.num_periods = self.X_setup.shape[1]
        self.num_samples = len(self.X_setup)
        
        print(f"Loaded {self.num_samples} samples with {self.num_periods} periods each")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Return a sample from the dataset.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            Tuple of (features, setup_target, production_target, inventory_target)
        """
        # Concatenate all features
        features = torch.cat([
            self.X_setup[idx],
            self.X_params[idx],
            self.X_demands[idx],
            self.X_init_inv[idx]
        ])
        
        # For multi-task learning, we'll return all targets
        return features, self.y_setup[idx], self.y_production[idx], self.y_inventory[idx]


def create_direct_loaders(npz_file, batch_size=64):
    """Create data loaders directly from a .npz file.
    
    This is a helper function to handle cases where the file doesn't exist
    in the expected package structure.
    
    Args:
        npz_file: Path to the .npz file
        batch_size: Batch size for the data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    # Load dataset directly
    dataset = ProductionPlanningDataset(npz_file)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, dataset


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, setup_weight=1.0, 
               production_weight=0.5, inventory_weight=0.5):
    """Train the neural network model.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use for training
        epochs: Number of epochs
        lr: Learning rate
        setup_weight: Weight for setup loss
        production_weight: Weight for production loss
        inventory_weight: Weight for inventory loss
        
    Returns:
        Tuple of (trained_model, train_losses, val_losses)
    """
    # Setup loss functions
    setup_criterion = nn.BCELoss()
    production_criterion = nn.MSELoss()
    inventory_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training on device: {device}")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, y_setup, y_production, y_inventory in train_loader:
            features = features.to(device)
            y_setup = y_setup.to(device)
            y_production = y_production.to(device)
            y_inventory = y_inventory.to(device)
            
            optimizer.zero_grad()
            
            setup_pred, production_pred, inventory_pred = model(features)
            
            # Calculate individual losses
            setup_loss = setup_criterion(setup_pred, y_setup)
            production_loss = production_criterion(production_pred, y_production)
            inventory_loss = inventory_criterion(inventory_pred, y_inventory)
            
            # Combined loss with weighting
            loss = (
                setup_weight * setup_loss + 
                production_weight * production_loss + 
                inventory_weight * inventory_loss
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        setup_preds = []
        setup_trues = []
        
        with torch.no_grad():
            for features, y_setup, y_production, y_inventory in val_loader:
                features = features.to(device)
                y_setup = y_setup.to(device)
                y_production = y_production.to(device)
                y_inventory = y_inventory.to(device)
                
                setup_pred, production_pred, inventory_pred = model(features)
                
                # Calculate individual losses
                setup_loss = setup_criterion(setup_pred, y_setup)
                production_loss = production_criterion(production_pred, y_production)
                inventory_loss = inventory_criterion(inventory_pred, y_inventory)
                
                # Combined loss with weighting
                loss = (
                    setup_weight * setup_loss + 
                    production_weight * production_loss + 
                    inventory_weight * inventory_loss
                )
                
                val_loss += loss.item()
                
                # Collect predictions for metrics
                setup_preds.append((setup_pred > 0.5).float().cpu().numpy())
                setup_trues.append(y_setup.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Calculate metrics for setup decisions
        setup_preds = np.vstack(setup_preds)
        setup_trues = np.vstack(setup_trues)
        accuracy = accuracy_score(setup_trues.flatten(), setup_preds.flatten())
        precision = precision_score(setup_trues.flatten(), setup_preds.flatten(), zero_division=0)
        recall = recall_score(setup_trues.flatten(), setup_preds.flatten(), zero_division=0)
        f1 = f1_score(setup_trues.flatten(), setup_preds.flatten(), zero_division=0)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Setup Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    setup_preds = []
    setup_trues = []
    production_preds = []
    production_trues = []
    inventory_preds = []
    inventory_trues = []
    
    with torch.no_grad():
        for features, y_setup, y_production, y_inventory in test_loader:
            features = features.to(device)
            
            setup_pred, production_pred, inventory_pred = model(features)
            
            # Convert setup predictions to binary
            setup_pred = (setup_pred > 0.5).float()
            
            # Collect predictions and true values
            setup_preds.append(setup_pred.cpu().numpy())
            setup_trues.append(y_setup.cpu().numpy())
            production_preds.append(production_pred.cpu().numpy())
            production_trues.append(y_production.cpu().numpy())
            inventory_preds.append(inventory_pred.cpu().numpy())
            inventory_trues.append(y_inventory.cpu().numpy())
    
    # Convert to numpy arrays
    setup_preds = np.vstack(setup_preds)
    setup_trues = np.vstack(setup_trues)
    production_preds = np.vstack(production_preds)
    production_trues = np.vstack(production_trues)
    inventory_preds = np.vstack(inventory_preds)
    inventory_trues = np.vstack(inventory_trues)
    
    # Calculate metrics for setup decisions
    setup_accuracy = accuracy_score(setup_trues.flatten(), setup_preds.flatten())
    setup_precision = precision_score(setup_trues.flatten(), setup_preds.flatten(), zero_division=0)
    setup_recall = recall_score(setup_trues.flatten(), setup_preds.flatten(), zero_division=0)
    setup_f1 = f1_score(setup_trues.flatten(), setup_preds.flatten(), zero_division=0)
    
    # Calculate MSE for production and inventory
    production_mse = np.mean((production_preds - production_trues) ** 2)
    inventory_mse = np.mean((inventory_preds - inventory_trues) ** 2)
    
    # Calculate normalized MSE (divide by mean of true values)
    production_nmse = production_mse / np.mean(production_trues ** 2) if np.mean(production_trues ** 2) > 0 else float('inf')
    inventory_nmse = inventory_mse / np.mean(inventory_trues ** 2) if np.mean(inventory_trues ** 2) > 0 else float('inf')
    
    print("\nTest Results:")
    print(f"Setup Decision Metrics:")
    print(f"  Accuracy: {setup_accuracy:.4f}")
    print(f"  Precision: {setup_precision:.4f}")
    print(f"  Recall: {setup_recall:.4f}")
    print(f"  F1 Score: {setup_f1:.4f}")
    print(f"Production Metrics:")
    print(f"  MSE: {production_mse:.4f}")
    print(f"  NMSE: {production_nmse:.4f}")
    print(f"Inventory Metrics:")
    print(f"  MSE: {inventory_mse:.4f}")
    print(f"  NMSE: {inventory_nmse:.4f}")
    
    return {
        'setup_accuracy': setup_accuracy,
        'setup_precision': setup_precision,
        'setup_recall': setup_recall,
        'setup_f1': setup_f1,
        'production_mse': production_mse,
        'production_nmse': production_nmse,
        'inventory_mse': inventory_mse,
        'inventory_nmse': inventory_nmse
    }

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train production planning neural network')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--setup_weight', type=float, default=1.0, help='Weight for setup loss')
    parser.add_argument('--production_weight', type=float, default=0.5, help='Weight for production loss')
    parser.add_argument('--inventory_weight', type=float, default=0.5, help='Weight for inventory loss')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    
    args = parser.parse_args()
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all block data files
    block_files = [f for f in os.listdir(args.data_dir) if f.startswith("production_planning_ml_data_block") and f.endswith(".npz")]
    
    if not block_files:
        print("No data files found!")
        return
    
    print(f"Found {len(block_files)} data files: {block_files}")
    
    # Train a model for each block size
    for file in block_files:
        print(f"\nTraining model for {file}")
        
        # Create data loaders directly since we're dealing with raw files
        train_loader, val_loader, test_loader, dataset = create_direct_loaders(
            os.path.join(args.data_dir, file),
            batch_size=args.batch_size
        )
        
        # Calculate input dimension
        sample_features, _, _, _ = dataset[0]
        input_dim = len(sample_features)
        
        # Initialize model
        model = ProductionPlanningNN(
            input_dim=input_dim,
            num_periods=dataset.num_periods,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            setup_weight=args.setup_weight,
            production_weight=args.production_weight,
            inventory_weight=args.inventory_weight
        )
        
        # Plot training history
        plot_training_history(
            train_losses, 
            val_losses,
            save_path=os.path.join(args.output_dir, f"training_history_{os.path.basename(file).replace('.npz', '.png')}")
        )
        
        # Evaluate model
        metrics = evaluate_model(trained_model, test_loader, device)
        
        # Save model
        model_save_path = os.path.join(args.output_dir, f"production_planning_model_{os.path.basename(file).replace('.npz', '.pt')}")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'input_dim': input_dim,
            'num_periods': dataset.num_periods,
            'hidden_dim': args.hidden_dim,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, model_save_path)
        
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()