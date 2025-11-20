# Decoder implementation for Dreamer
# Reconstructs spectrograms from latent states

import torch
import torch.nn as nn

from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

_logger = get_logger("decoder", level="DEBUG")


class Decoder(nn.Module):
    """Decoder for reconstructing spectrograms from latent states
    
    Args:
        h_state_size: Size of deterministic state (default: 200)
        z_state_size: Size of stochastic state (default: 30)
        out_channels: Output channels (default: 1 for mono audio)
        cnn_depth: Base depth for convolutional layers (default: 32)
    """
    
    def __init__(self,
                 h_state_size: int = 200,
                 z_state_size: int = 30,
                 out_channels: int = 1,
                 cnn_depth: int = 32):
        super().__init__()
        
        self.cnn_depth = cnn_depth
        
        # MLP to expand latent state
        # Initial spatial dimensions after encoder: (4, 3) for input (64, 50)
        # This should match the encoder's output spatial dimensions
        latent_size = h_state_size + z_state_size
        mlp_output_size = cnn_depth * 8 * 4 * 3  # (cnn_depth * 8 channels, 4 height, 3 width)
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ELU(),
            nn.Linear(200, mlp_output_size)
        )
        
        # Transposed convolutions to upsample
        # With padding=1, kernel=4, stride=2: out = (in - 1) * stride - 2*padding + kernel + output_padding
        # Working backwards from (64, 50): need (4,3) → (8,6) → (16,12) → (32,25) → (64,50)
        # output_padding=(0,1) only on layer 3 to get exact dimensions
        d = cnn_depth
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d * 8, d * 4, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(d * 4, d * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(d * 2, d, kernel_size=4, stride=2, padding=1, output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(d, out_channels, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, h_state, z_state):
        """
        Decode latent states to spectrograms
        
        Args:
            h_state: Deterministic state (B, T, h_state_size)
            z_state: Stochastic state (B, T, z_state_size)
            
        Returns:
            Reconstructed spectrograms (B, T, C, H, W)
        """
        # Concatenate states
        latent = torch.cat([h_state, z_state], dim=-1)
        
        _logger.debug(f"Latent shape: {latent.shape}")
        
        # Flatten batch and time
        latent, batch_dim = flatten_batch(latent)
        _logger.debug(f"Flattened latent shape: {latent.shape}")
        
        # MLP
        x = self.mlp(latent)
        _logger.debug(f"After MLP: {x.shape}")
        
        # Reshape to feature map (cnn_depth * 8 channels, 4x3 spatial)
        x = x.view(-1, self.cnn_depth * 8, 4, 3)
        _logger.debug(f"Reshaped: {x.shape}")
        
        # Deconvolution
        x = self.deconv(x)
        _logger.debug(f"After deconv: {x.shape}")
        
        # Unflatten batch
        x = unflatten_batch(x, batch_dim)
        _logger.debug(f"Final output: {x.shape}")
        
        return x
