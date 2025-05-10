import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head attention layer for capturing temporal relationships."""
    
    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Apply linear transformation and reshape
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        self.scale = self.scale.to(Q.device)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.out(out)
        
        return out, attention

class ProductionPlanningNN(nn.Module):
    """Neural network model for production planning with attention and regularization techniques.
    
    This model uses a shared base network followed by attention layers, then three task-specific heads,
    with added dropout, L2 regularization, and attention for capturing temporal dependencies.
    """
    
    def __init__(self, input_dim, num_periods, hidden_dim=64, dropout_rate=0.3, num_attention_heads=4, 
                 return_attention=False):
        """Initialize the neural network with attention and regularization features.
        
        Args:
            input_dim: Dimension of the input features
            num_periods: Number of time periods in the planning horizon
            hidden_dim: Dimension of hidden layers (reduced from original 128)
            dropout_rate: Rate for dropout layers
            num_attention_heads: Number of heads in multi-head attention
            return_attention: Whether to return attention weights (for backward compatibility)
        """
        super(ProductionPlanningNN, self).__init__()
        self.num_periods = num_periods
        self.hidden_dim = hidden_dim
        self.return_attention = return_attention
        
        # Common base layers with dropout
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Feature expansion to create sequence for attention
        self.feature_expansion = nn.Linear(hidden_dim, hidden_dim * num_periods)
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(hidden_dim, num_attention_heads, dropout_rate)
        
        # Attention normalization and residual connection
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        # Cross-attention layers for interaction between features
        self.cross_attention = MultiHeadAttention(hidden_dim, num_attention_heads, dropout_rate)
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward layer after attention
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        
        # Task-specific heads with attention-enhanced features
        self.setup_head = nn.Sequential(
            nn.Linear(hidden_dim * num_periods, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_periods),
            nn.Sigmoid()  # For binary classification
        )
        
        self.production_head = nn.Sequential(
            nn.Linear(hidden_dim * num_periods, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_periods),
            nn.ReLU()  # Production is non-negative
        )
        
        self.inventory_head = nn.Sequential(
            nn.Linear(hidden_dim * num_periods, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_periods),
            nn.ReLU()  # Inventory is typically non-negative
        )
    
    def forward(self, x):
        """Forward pass through the network with attention.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of outputs - either (setup, production, inventory) or 
            (setup, production, inventory, attention_weights) based on return_attention flag
        """
        # Base feature extraction
        base_features = self.base(x)  # [batch_size, hidden_dim]
        
        # Expand features to create a sequence for attention
        expanded_features = self.feature_expansion(base_features)  # [batch_size, hidden_dim * num_periods]
        sequence_features = expanded_features.view(x.size(0), self.num_periods, self.hidden_dim)  # [batch_size, num_periods, hidden_dim]
        
        # Self-attention on temporal features
        attended_features, attention_weights = self.attention(sequence_features, sequence_features, sequence_features)
        attended_features = self.attention_norm(sequence_features + self.attention_dropout(attended_features))
        
        # Cross-attention between attended features
        cross_attended_features, cross_attention_weights = self.cross_attention(attended_features, attended_features, attended_features)
        cross_attended_features = self.cross_attention_norm(attended_features + self.attention_dropout(cross_attended_features))
        
        # Feed-forward layer with residual connection
        ff_output = self.ff(cross_attended_features)
        ff_output = self.ff_norm(cross_attended_features + self.attention_dropout(ff_output))
        
        # Flatten attention output for task heads
        flattened_features = ff_output.view(x.size(0), -1)  # [batch_size, hidden_dim * num_periods]
        
        # Task-specific outputs
        setup_output = self.setup_head(flattened_features)
        production_output = self.production_head(flattened_features)
        inventory_output = self.inventory_head(flattened_features)
        
        if self.return_attention:
            return setup_output, production_output, inventory_output, attention_weights
        else:
            return setup_output, production_output, inventory_output

    def get_l2_regularization_loss(self, weight_decay=0.01):
        """Calculate L2 regularization loss for the model parameters.
        
        Args:
            weight_decay: Weight decay factor for L2 regularization
            
        Returns:
            Tensor containing the L2 regularization loss
        """
        l2_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        return weight_decay * l2_reg
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization and interpretation.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary containing attention weights for different attention layers
        """
        # Temporarily enable attention output
        original_return_attention = self.return_attention
        self.return_attention = True
        
        with torch.no_grad():
            _, _, _, attention_weights = self.forward(x)
            
            # Store attention weights for analysis
            attention_results = {
                'self_attention': attention_weights.cpu().numpy(),
                'averaged_attention': attention_weights.mean(dim=1).cpu().numpy()  # Average across heads
            }
        
        # Restore original setting
        self.return_attention = original_return_attention
        return attention_results

# Example usage and visualization helper
def visualize_attention(model, input_data, period_labels=None):
    """Helper function to visualize attention weights.
    
    Args:
        model: The ProductionPlanningNN model
        input_data: Input tensor for the model
        period_labels: Optional labels for time periods
        
    Returns:
        Dictionary containing attention visualization data
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    attention_weights = model.get_attention_weights(input_data)
    
    # Create visualization for averaged attention
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights['averaged_attention'][0], 
                cmap='viridis', 
                xticklabels=period_labels, 
                yticklabels=period_labels,
                annot=True, 
                fmt='.3f')
    plt.title('Attention Weights Between Time Periods')
    plt.xlabel('Key Periods')
    plt.ylabel('Query Periods')
    plt.tight_layout()
    
    return attention_weights