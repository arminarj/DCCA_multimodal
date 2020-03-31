import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        self.input_size = input_size ## flatting the data.
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1], bias=True),
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1], bias=True),
                    nn.ReLU(),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device='cpu'):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the Matrix needs to be make correlated X.shape= seq x features
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
