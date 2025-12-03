import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Compute required padding to maintain same T_out = T
        self.causal_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,  # No downsampling to maintain T_out = T
            dilation=dilation,
            padding=self.causal_padding,
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) - Batch, Time, Features

        Returns:
            (B, T, F) - Output sequence with same time steps
        """
        x = x.permute(0, 2, 1)  # (B, D, T) for Conv1d
        x = self.conv(x)

        # Remove extra padding on the right to ensure causality
        x = x[:, :, : -self.causal_padding] if self.causal_padding > 0 else x

        x = x.permute(0, 2, 1)  # Back to (B, T, F)
        return x
