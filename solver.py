import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

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
import os
import time

from dataset import *
import torch.optim as optim


class Solver():
    def __init__(self, model, linear_cca, outdim_size1, outdim_size2, hyp_params):
        # self.model = nn.DataParallel(model)
        self.model = model.double()
        self.model.to(hyp_params.device)
        self.epoch_num = hyp_params.num_epochs
        self.batch_size = hyp_params.batch_size
        self.batch_chunk = hyp_params.batch_size
        self.loss = model.loss
        self.optimizer = getattr(optim, hyp_params.optim)(
            self.model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.reg_par)
        self.device = hyp_params.device
        self.schedule = ReduceLROnPlateau(self.optimizer, mode='min', patience=hyp_params.when,
                                             factor=0.1, verbose=True)
        self.log_interval = hyp_params.log_interval
        self.use_linear_cca = True
        self.linear_cca = linear_cca

        self.outdim_size1 = outdim_size1
        self.outdim_size2 = outdim_size2

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
        self.writer = SummaryWriter()
        self.epoch = -1

    def fit(self, train_loader, valid_loader, test_loader, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, seq, feats]

        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        best_valid = 1e+08
        for epoch in range(1, self.epoch_num+1):
            self.epoch = epoch
            start = time.time()
            train_loss = self.train(epoch)
            val_loss, _ = self.test(False)
            test_loss, _ = self.test(True) 
            end = time.time()
            duration = end-start
            self.schedule.step(val_loss)    # Decay learning rate by validation loss

            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, train_loss, val_loss, test_loss))
            print("-"*50)

            if val_loss < best_valid:
                name = self.hyp_params.name
                save_model(self.hyp_params, self.model, name=name)
                print(f"Saved model at pre_trained_models/{name}.pt")
                best_valid = val_loss
                
        self.model = load_model(self.hyp_params, name=self.hyp_params.name)

        if self.linear_cca is not None:
            self._train_linear_dcca()
    
        
        self.writer.close()

        return  self.model
        

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
        batch_size = self.batch_size
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(self.train_loader):
            sample_ind, modal1, modal2 = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()

            modal1, modal2 = modal1.to(self.device).double(), modal2.to(self.device).double()
            batch_chunk = modal1.size(0)
                
            combined_loss = 0
            net = nn.DataParallel(model) if self.batch_size > 10 else model
            # net = model
        

            raw_loss = combined_loss = 0
            modal1, modal2 = modal1.unsqueeze(1), modal2.unsqueeze(1)
            o1, o2 = net(modal1, modal2)
            raw_loss_i = criterion(o1, o2)
            raw_loss += raw_loss_i
            raw_loss_i.backward()
            combined_loss = raw_loss 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.hyp_params.clip)

            optimizer.step()

            
            proc_loss += raw_loss.item() * batch_chunk
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_chunk
            if i_batch % self.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                    format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                self.writer.add_graph(self.model, (modal1, modal2))
                

        for name, param in self.model.model1.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.epoch)
        for name, param in self.model.model1.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.epoch)
        self.writer.add_scalar('Loss/train', avg_loss, self.epoch)
        return avg_loss

    def evaluate(self, test=False, _loader=None):
        model=self.model.to(self.device)
        device = self.device
        criterion=self.loss    
        model.eval()
        if _loader is None:
            loader = self.test_loader if test else self.valid_loader
        else : loader = _loader
        total_loss = 0.0
        batch_size = self.batch_size 

        net = model
        output1, output2, losses = [], [], []
        labels = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                _, modal1, modal2= batch_X
                modal1, modal2 = modal1.to(device).double(), modal2.to(device).double()
                batch_chunk = modal1.size(0)
                eval_attr = batch_Y.squeeze(-1)
                modal1, modal2 = modal1.unsqueeze(1), modal2.unsqueeze(1)
                o1, o2 = net(modal1, modal2)
                output1.append(o1)
                output2.append(o2)
                
                raw_loss = criterion(o1, o2)
                total_loss += raw_loss.item() 
                losses.append(total_loss)
                labels.append(eval_attr)

            
            labels = torch.cat(labels, dim=0)
            outputs = [torch.cat(output1, dim=0),
                        torch.cat(output2, dim=0)]
            losses = torch.tensor(losses).double().mean()
        
        if loader is self.train_loader:
            loss_name="train_eval"
        elif test:
            loss_name='test_eval'
        else : loss_name ='valid_eval'
        mat = np.concatenate([
                outputs[0].cpu().numpy(),
                outputs[1].cpu().numpy(),
            ], axis=1)
        labels = np.concatenate([labels.data,labels.data], axis=1)
        # self.writer.add_embedding(mat,
        #                         metadata=labels,
        #                         global_step=self.epoch
        #                         # tag=["text", "audio"]
        #                         )

        self.writer.add_scalar(f'Loss/{loss_name}', losses, self.epoch)

        outputs[0], outputs[1] = outputs[0].to(self.device), outputs[1].to(self.device)
        # assert len(outputs[0]) != (50*24*678)
        return losses, outputs

    def test(self, test=False, loader=None):
        losses, outputs = self.evaluate(test=test, _loader=loader)
        if self.use_linear_cca:
            print("Linear CCA started!")
            outputs = self.linear_cca.test(outputs[0], outputs[1])
            return losses, outputs
        else:
            return losses, outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size1, self.outdim_size2)

    def _train_linear_dcca(self):
        print(f'Start linear CCA Training...')
        _, outputs = self.test(loader=self.train_loader)
        self.train_linear_cca(outputs[0], outputs[1])
        print(f'End linear CCA Training...')