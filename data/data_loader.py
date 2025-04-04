import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class ProductionPlanningDataset(Dataset):
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


def create_data_loaders(npz_file, batch_size=64, train_ratio=0.7, val_ratio=0.15, 
                        test_ratio=0.15, shuffle=True):
    """Create DataLoaders for training, validation, and testing.
    
    Args:
        npz_file: Path to the .npz file containing training data
        batch_size: Batch size for the DataLoaders
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        shuffle: Whether to shuffle the training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset)
    """
    dataset = ProductionPlanningDataset(npz_file)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, dataset