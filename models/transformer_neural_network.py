import torch
import torch.nn as nn

class TransformerProductionPlanningNN(nn.Module):
    """Transformer-based neural network model for production planning.
    
    This model uses a transformer architecture with linear feature embeddings,
    followed by task-specific heads for setup, production, and inventory prediction.
    """
    
    def __init__(self, input_dim, num_periods, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """Initialize the transformer-based neural network.
        
        Args:
            input_dim: Dimension of the input features
            num_periods: Number of time periods in the planning horizon
            hidden_dim: Dimension of hidden layers and transformer model
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate for transformer
        """
        super(TransformerProductionPlanningNN, self).__init__()
        self.num_periods = num_periods
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding is not strictly necessary since our inputs aren't sequential,
        # but we include it for completeness
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
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
        # Reshape input to add sequence length dimension if it's not already present
        if len(x.shape) == 2:
            # [batch_size, input_dim] -> [batch_size, 1, input_dim]
            x = x.unsqueeze(1)
        
        # Linear feature embedding
        x = self.feature_embedding(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output of the first (and only) token
        # or average across all tokens if there are multiple
        if x.shape[1] == 1:
            x = x.squeeze(1)  # [batch_size, hidden_dim]
        else:
            x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        # Apply task-specific heads
        setup_output = self.setup_head(x)
        production_output = self.production_head(x)
        inventory_output = self.inventory_head(x)
        
        return setup_output, production_output, inventory_output


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    
    def __init__(self, d_model, dropout=0.1, max_len=100):
        """Initialize the positional encoding.
        
        Args:
            d_model: Dimension of the model
            dropout: Dropout rate
            max_len: Maximum length of the input sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Forward pass through the positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

