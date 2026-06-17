import torch
import torch.nn as nn
import types
import inspect
from transformers import PreTrainedModel

class ParallelAdapter(nn.Module):
    """
    Parallel adapter that receives pre-computed embeddings from an external source.
    """
    def __init__(self, hidden_size, adapter_type="linear", rank=8, scaling=1.0):
        super().__init__()
        self.adapter_type = adapter_type
        self.hidden_size = hidden_size
        self.scaling = scaling
        
        if adapter_type == "linear":
            # Single matrix linear transform
            self.adapter = nn.Linear(hidden_size, hidden_size, bias=False)
            # Initialize to zero for residual connection at training start
            nn.init.zeros_(self.adapter.weight)

        elif adapter_type == "affine":
            # Single matrix affine transform
            self.adapter = nn.Linear(hidden_size, hidden_size, bias=True)
            # Initialize to zero for stable training start
            nn.init.zeros_(self.adapter.bias)
            nn.init.zeros_(self.adapter.weight)
        
        elif adapter_type == "low_rank":
            # Low-rank implementation (two matrices multiplication)
            self.down_proj = nn.Linear(hidden_size, rank, bias=False)
            self.up_proj = nn.Linear(rank, hidden_size, bias=False)
            
            # Initialize with small random weights for down, zeros for up
            nn.init.kaiming_uniform_(self.down_proj.weight)
            nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, embeddings):
        """
        Forward pass using pre-computed embeddings
        
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, hidden_size]
                       containing pre-computed embeddings
        """
        if self.adapter_type == "linear":
            return self.adapter(embeddings) * self.scaling
        elif self.adapter_type == "affine":
            return self.adapter(embeddings) * self.scaling
        elif self.adapter_type == "low_rank":
            return self.up_proj(self.down_proj(embeddings)) * self.scaling
        else:
            raise ValueError(f"Unsupported adapter type '{self.adapter_type}' in forward pass")
