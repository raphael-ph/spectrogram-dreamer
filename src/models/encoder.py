# For the original paper, here is the description of the Encoder Representation:
#
# >>> The representation model is implemented as a Convolutional Neural Network (CNN; LeCun et al., 1989) \
# >>> followed by a Multi-Layer Perceptron (MLP) that receives the image embedding and the deterministic recurrent state.
#
# For this encoder implementation, we will follow the Convolutional Neural Network, following the paper's parameters.
# We took inspiration from Pydreamer: http://github.com/jurgisp/pydreamer/blob/main/pydreamer/models/encoders.py

# Torch imports
import torch.nn as nn

# internal imports
from ..utils.logger import get_logger
from ..utils.functions import flatten_batch, unflatten_batch

# set up logging
_logger = get_logger("encoder", level="DEBUG")

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, cnn_depth: int = 32):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = (4, 4, 4, 4)
        stride = 2
        d = cnn_depth
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, d, kernels[0], stride),
            nn.ELU(), # paper specifies ELU as the activation function
            nn.Conv2d(d, d * 2, kernels[1], stride),
            nn.ELU(),
            nn.Conv2d(d * 2, d * 4, kernels[2], stride),
            nn.ELU(),
            nn.Conv2d(d * 4, d * 8, kernels[3], stride),
            nn.ELU(),
            nn.Flatten()
        )

    def forward(self, x):
        _logger.debug(f"x shape before flattening: {x.shape}") # expecting here (B, T, C, n_mels, L)
        
        x, batch_dim = flatten_batch(x, 3)
        _logger.debug(f"x shape after flattened: {x.shape}") # expecting here (B, X)

        y = self.model(x)
        _logger.debug(f"y shape flattened: {y.shape}")

        y = unflatten_batch(y, batch_dim)
        _logger.debug(f"y shape after unflattened: {y.shape}")

        return y