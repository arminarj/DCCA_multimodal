import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
import os 
import numpy as np 
import time

def get_data(args, dataset, split='train'):
    alignment = 'a' 
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=True):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        start_time = time.time()
        dataset = thepickle.load(open(dataset_path, 'rb'))
        print(f'end of reading file, total time {time.time()-start_time}')

        # These are torch tensors
        _vision_1 = 'OpenFace_2.0'
        _vision_2 = 'FACET 4.2'
        _audio_1 = 'COAVAREP' 
        _audio_2 = 'OpenSMILE'
        _text = 'glove_vectors' 
        _labels = 'All Labels'
        # ## OpenFace_2.0
        # self.vision = torch.tensor(dataset[split_type][_vision_1]).double().cpu().detach()
        # self.vision_1 = self.normalize(self.vision)
        # ## FACET 4.2
        # self.vision = torch.tensor(dataset[split_type][_vision_2]).double().cpu().detach()
        # self.vision_2 = self.normalize(self.vision)
        # ## glove_vectors
        # self.text = torch.tensor(dataset[split_type][_text]).double().cpu().detach()
        # self.text = self.normalize(self.text)
        ## COAVAREP 
        self.audio_1 = dataset[split_type][_audio_1]
        self.audio_1[self.audio_1 == -float("Inf")] = 0
        self.audio_1 = torch.tensor(self.audio_1).cpu().double().detach()
        self.audio_1 = self.normalize(self.audio_1)
        ## OpenSMILE 
        self.audio_2 = dataset[split_type][_audio_2].double()
        self.audio_2[self.audio_2 == -float("Inf")] = 0
        self.audio_2 = torch.tensor(self.audio_2).cpu().detach()
        self.audio_2 = self.normalize(self.audio_2)
        ## labels
        self.labels = torch.tensor(dataset[split_type][_labels]).double().cpu().detach()
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio

    def normalize(self, x): # input shape 50 x features
        _eps = 1e-8
        print(f'X shape is : {x.shape}')
        for index in range(x.shape[0]):
            for seq in range(x[index].shape[0]):
                x[index][seq] = (x[index][seq] - x[index][seq].min())/(x[index][seq].max()- x[index][seq].min() + _eps)
                # x[index][seq] = (x[index][seq] - 1/2) * 2
                assert torch.isnan(x[index][seq]).sum().item() == 0

        return x
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.audio_1.shape[1], self.audio_2.shape[1]
    def get_dim(self):
        return self.audio_1.shape[2], self.audio_2.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.audio_1[index], self.audio_2[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META       
