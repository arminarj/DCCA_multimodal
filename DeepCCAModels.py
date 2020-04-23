import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss
import torch.nn.functional as F


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size, p=0.3):
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
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                    nn.Dropout(p=p),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size1, outdim_size2,
                         use_all_singular_values, device='cpu', p=0.3):
        super(DeepCCA, self).__init__()
        # self.model1 = MlpNet(layer_sizes1, input_size1, p=p).double()
        # self.model2 = MlpNet(layer_sizes2, input_size2, p=p).double()
        self.device = device
        self.model1 = LeNet(input_size=input_size1, seq_size=50, output_size=outdim_size1, p=p).to(self.device).double()
        self.model2 = LeNet(input_size=input_size2, seq_size=50, output_size=outdim_size2, p=p).to(self.device).double()

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

class LeNet(nn.Module):
    def __init__(self, input_size=300, seq_size=50, output_size=512, p=0.3):
        super(LeNet, self).__init__()
        self.output_size = output_size
        self.kernel_size = 5
        self.fc_shape1 = self.fc_shape(input_size)
        self.fc_shape2 = self.fc_shape(seq_size)
        self.conv1 = nn.Conv2d(1, 6, self.kernel_size)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.p = p
        print(16*self.fc_shape1*self.fc_shape2)
        self.fc   = nn.Sequential(
            nn.Linear(16*self.fc_shape1*self.fc_shape2, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.BatchNorm1d(num_features=1024, affine=False),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.BatchNorm1d(num_features=1024, affine=False),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Dropout(p=self.p),
            nn.BatchNorm1d(num_features=512, affine=False),
            nn.Linear(512, self.output_size),
        )
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def fc_shape(self, in_size):
        fc_shape = int((in_size-self.kernel_size+1-2)/2 + 1)
        fc_shape = int((fc_shape-self.kernel_size+1-2)/2 + 1)
        if fc_shape <= 0 :
            print(fc_shape)
            assert 'LeNet wrong calculator'
        return fc_shape