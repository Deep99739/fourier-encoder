import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjection(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(embedding_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            # self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(dropout))
        
        # Final layer to map back to embedding dimension
        self.final_layer = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass through the intermediate network.
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        # Process each position independently
        x_orig = x
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.view(-1, self.embedding_dim)
        
        # Pass through layers
        for layer in self.layers:
            x_reshaped = layer(x_reshaped)
        
        # Final layer to map back to embedding dimension
        x_reshaped = self.final_layer(x_reshaped)
        
        # Reshape back to original dimensions
        x = x_reshaped.view(batch_size, seq_len, self.embedding_dim)
        return x + x_orig
    

class LinearProjection(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=2048, num_layers=2, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        # Create a single linear layer that maps from embedding_dim to embedding_dim
        self.final_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.final_layer(x)
        return x
    

class IdentityProjection(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=2048, num_layers=2, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
    
    def forward(self, x):
        return x

# class NFProjection(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim=2048, num_layers=2, dropout=0.1, device='cuda'):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.device = device
    
#     def forward(self, x):
#         return x