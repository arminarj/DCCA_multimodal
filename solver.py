import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
import os
import time

from dataset import *


class Solver():
    def __init__(self, model, linear_cca, outdim_size, hyp_params):
        # self.model = nn.DataParallel(model)
        self.model = model.double()
        self.model.to(hyp_params.device)
        self.epoch_num = hyp_params.num_epochs
        self.batch_size = hyp_params.batch_size
        self.batch_chunk = hyp_params.batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.reg_par)
        self.device = hyp_params.device
        self.schedule = ReduceLROnPlateau(self.optimizer, mode='min', patience=hyp_params.when,
                                             factor=0.1, verbose=True)
        self.log_interval = hyp_params.log_interval
        self.use_linear_cca = True
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

        for epoch in range(1, self.epoch_num+1):
            start = time.time()
            self.train(epoch)
            val_loss, _ = self.test(False)
            test_loss, _ = self.test(True)
            
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

        ####################################
        ###### functions defenition #######
        ####################################

    def train(self, epoch):
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

            text, audio = text.to(self.device).double(), audio.to(self.device).double()
            batch_size = text.size(0)
            batch_chunk = self.batch_chunk
                
            combined_loss = 0
            # net = nn.DataParallel(model) if batch_size > 10 else model
            net = model
        
            if self.batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                # vision_chunks = vision.chunk(batch_chunk, dim=0)
                # eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(self.batch_chunk):
                    text_i, audio_i = text_chunks[i], audio_chunks[i]
                    print(f'text is : {text_i[0].mean()}')
                    print(f'audio is : {audio_i[0].mean()}')
                    # if i is 0:
                    #     print (f'text, audio max : {text_i.max()}, {audio_i.max()}')
                    #     print (f'text, audio min : {text_i.min()}, {audio_i.min()}') 
                        # print (f'bias : {net.model1.layers[0][0].bias}')
                    # print (f'b grad : {net.model1.layers[0][0].bias.grad}')
                    # print (f'w grad : {net.model1.layers[0][0].weight.grad}')
                    o1, o2 = net(text_i, audio_i)
                    # print(f'output shape : {o1.shape}')
                    o1, o2 = o1.squeeze(), o2.squeeze()
                    # print(f'grad of raw loss : {o1.grad}')
                    raw_loss_i = criterion(o1, o2)
                    raw_loss += raw_loss_i
                raw_loss = raw_loss 
                raw_loss.backward()
                print('backward')
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
        if self.linear_cca is not None:
            _, outputs = self.test(loader=train_loader)
            self.train_linear_cca(outputs[0], outputs[1])
                
        return epoch_loss / self.hyp_params.n_train

    def evaluate(self, test=False, _loader=None):
        model=self.model
        device = self.device
        criterion=self.loss    
        model.eval()
        if _loader is None:
            loader = self.test_loader if test else self.valid_loader
        else : loader = _loader
        total_loss = 0.0
    

        net = model
        output1, output2, losses = [], [], []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                _, text, audio, _ = batch_X
                text, audio = text.to(self.device).double(), audio.to(self.device).double()
                batch_size = text.size(0)
                batch_chunk = self.batch_chunk

                if batch_chunk > 1:
                    raw_loss = combined_loss = 0
                    text_chunks = text.chunk(batch_chunk, dim=0)
                    audio_chunks = audio.chunk(batch_chunk, dim=0)
                    # vision_chunks = vision.chunk(batch_chunk, dim=0)
                    
                    for i in range(self.batch_chunk):
                        text_i, audio_i = text_chunks[i], audio_chunks[i]
                        # if i is 0:
                            # print (f'text, audio max : {text_i.max()}, {audio_i.max()}')
                            # print (f'text, audio min : {text_i.min()}, {audio_i.min()}') 
                        o1, o2 = net(text_i, audio_i)
                        o1, o2 = o1.squeeze(), o2.squeeze()
                        output1.append(o1)
                        output2.append(o2)
                        raw_loss_i = criterion(o1, o2)
                        # print(f'raw loss : {raw_loss_i}')
                        raw_loss += raw_loss_i
                        total_loss += criterion(o1, o2).item() * batch_size
                    losses.append(total_loss)
                else:
                    o1, o2 = net(text, audio)
                    output1.append(o1)
                    output2.append(o2)
                    raw_loss = self.loss(o1, o2)
                    combined_loss = raw_loss
                    losses.append(total_loss) 
                
            outputs = [torch.cat(output1, dim=0),
                        torch.cat(output2, dim=0)]
        return losses, outputs

    def test(self, test=False, loader=None):
        losses, outputs = self.evaluate(test=test, _loader=loader)
        if self.use_linear_cca:
            print("Linear CCA started!")
            outputs = self.linear_cca.test(outputs[0], outputs[1])
            return torch.mean(losses), outputs
        else:
            return torch.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)
