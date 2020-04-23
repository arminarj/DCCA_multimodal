%load_ext autoreload
%autoreload 2
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim

import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import *
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
import os
import time

from dataset import *

from solver import Solver
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)


parser = argparse.ArgumentParser(description='MOSEI DeepCCA Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='DeepCCA',
                    help='name of the model to use (Transformer, etc.)')

parser.add_argument('--aligned', default=True,
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='torch_tensors_audio',
                    help='dataset to use (default: "torch_tensors_audio" ')
parser.add_argument('--data_path', type=str, default='/content/DCCA_multimodal/data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--dropout', type=float, default=0.05,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')



# Tuning
parser.add_argument('--batch_size', type=int, default=300, metavar='N',
                    help='batch size (default: 400)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-7,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs (default: 10)')
parser.add_argument('--when', type=int, default=5,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=50,
                    help='number of chunks per batch (default: 1)')
parser.add_argument('--reg_par', type=float, default=1e-8,
                    help='the regularization parameter of the network')
# Logistics
parser.add_argument('--log_interval', type=int, default=10,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='cca',
                    help='name of the trial (default: "cca")')
parser.add_argument('--nlevels', type=int, default=3,
                    help='n hidden layer')
args = parser.parse_args()


output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("Using", torch.cuda.device_count(), "GPUs")
        use_cuda = True

####################################################################
#
# Load the dataset 
#
####################################################################

print("Start loading the data....")
dataset = args.dataset

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

print('Finish loading the data....')


####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_1, hyp_params.orig_d_2 = train_data.get_dim()
hyp_params.d_1_len, hyp_params.d_2_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
# hyp_params.criterion = criterion_dict.get(dataset, 'NLLoss')
hyp_params.device = 'cuda' if use_cuda else 'cpu'



if __name__ == '__main__':
    ############
    # the size of the new space learned by the model (number of the new features)
    outdim_size1 = 10
    outdim_size2 = 10

    # size of the input for view 1 and view 2
    input_shape1 = hyp_params.orig_d_1
    input_shape2 = hyp_params.orig_d_2

    # number of layers with nodes in each one
    layer_size = [512]* (hyp_params.layers-1)
    layer_sizes1 = layer_size + [outdim_size1]
    layer_sizes2 = layer_size + [outdim_size2]

    # the parameters for training the network


    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
 
    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = True
    # end of parameters section
    ############


    # Building, training, and producing the new features by DCCA
    model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size1, outdim_size2, use_all_singular_values, device=hyp_params.device, p=hyp_params.dropout)

    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size1, outdim_size2, hyp_params)

    solver.fit(train_loader, valid_loader, test_loader)

