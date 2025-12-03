from typing import List
import torch.nn as nn

from cross_modal_distillation.utility.utils import get_activation_function


class MLP(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_out: int,
        layer_list: List = None,
        dropout: float = 0.1,
        use_final_dropout: bool = False,
        activation: str = "linear",
        **kwargs
    ):
        super(MLP, self).__init__()

        self.d_input = d_input
        self.d_out = d_out
        # Bookkeep d_out as d_hidden as well, we may need access to this attribute
        # across different backbones
        self.d_hidden = d_out
        self.layer_list = layer_list
        self.dropout = dropout
        self.use_final_dropout = use_final_dropout
        self.activation_fn = get_activation_function(activation)

        current_dim = self.d_input
        self.layers = nn.ModuleList()
        if self.layer_list is not None:
            for _, dim in enumerate(self.layer_list):
                self.layers.append(nn.Linear(current_dim, dim))
                current_dim = dim
        else:
            self.layers.append(nn.Identity())

        self.final_layer = nn.Linear(current_dim, self.d_out)

    def forward(self, x):
        x = nn.Dropout(self.dropout)(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
            x = nn.Dropout(self.dropout)(x)
        x = self.final_layer(x)
        if self.use_final_dropout:
            x = nn.Dropout(self.dropout)(x)
        return x
