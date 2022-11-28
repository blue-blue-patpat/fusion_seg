import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print(layer, x.shape)
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class PredictorLG(nn.Module):
    """ 
    From DydamicVit
    """
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.score_nets = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.score_nets(x)
        return x
    