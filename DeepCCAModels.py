import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss, gcca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size, p=0.1):
        super(MlpNet, self).__init__()
        self.input_size = input_size ## flatting the data.
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1], bias=True),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1], bias=True),
                    nn.ReLU(),
                    # nn.BatchNorm1d(num_features=layer_sizes[l_id + 1]),
                    # nn.Dropout(p=p),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size1, outdim_size2,
                         use_all_singular_values, device='cpu', p=0.1):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1, p=p).double()
        self.model2 = MlpNet(layer_sizes2, input_size2, p=p).double()

        self.loss = cca_loss(outdim_size1, outdim_size2, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the Matrix needs to be make correlated X.shape= seq x features
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2

class DeepGCCA(nn.Module):
    def __init__(self, layer_sizes, input_sizes, outdim_sizes,
                         use_all_singular_values, device='cpu', p=0.1):
        super(DeepGCCA, self).__init__()
        self.layer_sizes = layer_sizes
        self.input_sizes = input_sizes
        self.outdim_sizes = outdim_sizes
        self.model1 = MlpNet(layer_sizes[0], input_sizes[0], p=p).double()
        self.model2 = MlpNet(layer_sizes[1], input_sizes[1], p=p).double()
        self.model3 = MlpNet(layer_sizes[2], input_sizes[2], p=p).double()

        self.loss = gcca_loss(outdim_sizes, use_all_singular_values, device).loss

    def forward(self, x1, x2, x3):
        """

        x1, x2, x3 are the Matrix needs to be make correlated X.shape= seq x features
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        output3 = self.model3(x3)

        return output1, output2, output3
