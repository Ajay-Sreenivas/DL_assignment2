"""Reusable custom layers
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer using inverted dropout scaling.

    Inverted dropout multiplies surviving activations by 1/(1-p) during training
    so that the expected value of any activation is the same at test time (no
    scaling needed in eval mode).  When self.training is False the layer is a
    pure identity.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability (fraction of neurons to zero out).
               Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        During training:  samples a Bernoulli mask with success prob (1-p),
                          zeros out dropped activations and scales survivors
                          by 1/(1-p) (inverted dropout).
        During eval:      returns x unchanged.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Output tensor of the same shape.
        """
        if not self.training or self.p == 0.0:
            return x

        # Bernoulli mask: 1 with probability (1 - p), 0 with probability p
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        # Inverted dropout scaling so E[output] == x during training
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
