import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, svm_classify
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import os
import time

from dataset import *


class Solver():
    def __init__(self, model, linear_cca, outdim_size, schedule, hyp_params):
        self.model = nn.DataParallel(model)
        self.model.to(hyp_params.device)
        self.epoch_num = hyp_params.num_epochs
        self.batch_size = hyp_params.batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.reg_par)
        self.device = hyp_params.device
        self.schedule = schedule
        self.log_interval = 30

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

        self.hyp_params = hyp_params

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, train_loader, valid_loader, test_loader, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader

        ####################################
        #### end of functions defenition ###
        ####################################

        for epoch in range(1, self.epoch_num+1):
            start = time.time()
            self.train()
            val_loss, _ = test(test=False)
            test_loss, _ = test(test=True)
            
            end = time.time()
            duration = end-start
            self.scheduler.step(val_loss)    # Decay learning rate by validation loss

            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
            print("-"*50)
            
            if val_loss < best_valid:
                name = 'DCCA_AL'
                save_model(self.hyp_params, model, name=name)
                print(f"Saved model at pre_trained_models/{name}.pt")
                best_valid = val_loss

    def train(self):
            model=self.model
            optimizer=self.optimizer
            criterion=self.loss
            epoch_loss = 0
            device = self.device
            model.train()
            num_batches = self.hyp_params.n_train // self.batch_size
            proc_loss, proc_size = 0, 0
            start_time = time.time()
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(self.train_loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
                
                model.zero_grad()

                text, audio = text.to(device).to(torch.FloatTensor()), audio.to(device).to(torch.FloatTensor())
                batch_size = text.size(0)
                batch_chunk = batch_size
                    
                combined_loss = 0
                net = nn.DataParallel(model) if batch_size > 10 else model
            
                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    # vision_chunks = vision.chunk(batch_chunk, dim=0)
                    # eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                    
                    for i in range(batch_chunk):
                        text_i, audio_i = text_chunks[i], audio_chunks[i]
                        o1, o2 = net(text_i, audio_i)
                        raw_loss_i = criterion(o1, o2)
                        raw_loss += raw_loss_i
                        raw_loss_i.backward()
                    combined_loss = raw_loss 
                else:

                    o1, o2 = net(text, audio)
                    raw_loss = self.loss(o1, o2)
                    combined_loss = raw_loss 
                    combined_loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hyp_params.clip)
                optimizer.step()
                
                proc_loss += raw_loss.item() * batch_size
                proc_size += batch_size
                epoch_loss += combined_loss.item() * batch_size
                if i_batch % self.log_interval == 0 and i_batch > 0:
                    avg_loss = proc_loss / proc_size
                    elapsed_time = time.time() - start_time
                    print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                        format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hyp_params.log_interval, avg_loss))
                    proc_loss, proc_size = 0, 0
                    start_time = time.time()
                    
            return epoch_loss / self.hyp_params.n_train

    def __evaluate(self, test=False):
            model=self.model
            criterion=self.loss
            model.eval()
            loader = self.test_loader if test else self.valid_loader
            total_loss = 0.0
        

            net = model
            output1, output2, losses = [], [], []
            with torch.no_grad():
                for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                    _, text, audio, _ = batch_X
                    text, audio, _, _ = text.to(device).FloatTensor(), audio.to(device).FloatTensor()
                    batch_size = text.size(0)
                    o1, o2 = net[random_choosed](text, audio)
                    output1.append(o1)
                    output2.append(o2)
                    total_loss += criterion(o1, o2).item() * batch_size
                    losses.append(total_loss)
                    
                outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                            torch.cat(outputs2, dim=0).cpu().numpy()]
            return losses, outputs

    def test(test=False):
        losses, outputs = _evaluate(test=False)
        if use_linear_cca:
            print("Linear CCA started!")
            outputs = self.linear_cca.test(outputs[0], outputs[1])
            return np.mean(losses), outputs
        else:
            return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

