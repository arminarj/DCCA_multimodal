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
        dataset = thepickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.vision = self.normalize(self.vision)
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.text = self.normalize(self.text) 
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.audio = self.normalize(self.audio)
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio

    def normalize(self, x): # input shape 50 x features
        _eps = 1e-8
        print(f'X shape is : {x.shape}')
        for index in range(x.shape[0]):
            for seq in range(x[index].shape[0]):
                # if x[index][seq].max() == x[index][seq].min() and x[index][seq].max()==0:
                #     x[index][seq] = torch.zeros(x[index][seq])
                x[index][seq] = (x[index][seq] - x[index][seq].min())/(x[index][seq].max()- x[index][seq].min() + _eps)
                x[index][seq] = (x[index][seq] - 1/2) * 2
                assert torch.isnan(x[index][seq]).sum().item() == 0
        assert torch.isnan(x).sum().item() == 0
        return x
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        assert torch.isnan(self.text[index]).sum().item() == 0
        assert torch.isnan(self.audio[index]).sum().item() == 0
        assert torch.isnan(self.vision[index]).sum().item() == 0
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META        

