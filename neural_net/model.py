import torch
import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()

        layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Tanh())  # supports negative and positive values

        # Final layer (no activation, since it's a regression output)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
