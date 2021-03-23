import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from models import CNN_RNN, LeNet_1D
from utils.utils import asMinutes


class RegressionTrainer:
    """
    Training object class. Receive dedicated model and train it with train loader.
    :param net_type: model type we want to train
    :param net_param: model parameter
    :param device: gpu number
    :param optimizer: optimizer we want to use in training
    :param loss_fn: loss function we want to use in training
    :param optimizer_param: optimizer parameters
    :param loss_param: loss function parameters
    :param train_loader: train loader
    :param validation_loader: validation loader
    :param test_loader: test loader
    :param model_path: model save path
    :param loss_save_path: loss save path
    :param transpose: boolean to decide to transpose the output
    """

    def __init__(self, net_type, net_param, device, optimizer, loss_fn, optimizer_param, loss_param,
                 train_loader, validation_loader, test_loader, model_path, loss_save_path):
        _input_dim = 18
        if net_type == 'CNN_RNN':
            self.net = CNN_RNN(**net_param, device=device)
        else:
            self.net = LeNet_1D(**net_param)
        self.device = device
        self.optimizer = getattr(optim, optimizer)(self.net.parameters(), **optimizer_param)

        try:
            self.loss_fn = getattr(nn, loss_fn)(**loss_param)
        except:
            print("Oops! No such loss function in torch.nn package")

        if hasattr(nn, loss_fn):
            self.loss_fn = getattr(nn, loss_fn)(**loss_param)
        else:
            self.loss_fn = loss_fn()
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.model_path = model_path
        self.loss_save_path = loss_save_path

        self.train_loss = []
        self.valid_loss = []

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_loader)
        self.train_loss.append(loss)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_loader, train=False)
        self.valid_loss.append(loss)
        return loss

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over data loader for train or test.
        :param epoch: current epoch idx
        :param data_loader: data loader (train, test, valid)
        :param train: boolean value for train or not
        :return: None
        """
        lossSum = 0
        epoch_time = time.time()

        if train:
            self.net.train()
        else:
            self.net.eval()

        for idx, (X, y) in enumerate(data_loader):
            X = torch.reshape(X, (X.shape[0], 128, -1))
            X = X.permute(0, 2, 1)
            X.to(self.device)
            y.to(self.device)

            out = self.net(X)

            loss = self.loss_fn(out, y)

            lossSum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if train:
            print('Epoch{} took : {}, avg_loss : {:.4f}'.format(epoch, asMinutes(time.time() - epoch_time),
                                                                lossSum / (idx + 1)))
        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
        #       total_correct * 100.0 / total_element)

        return lossSum / (idx + 1)

    def save(self, epoch, file_name="model"):
        self.net.cpu()
        state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(self.model_path, '%s_ep%d.pt' % (file_name, epoch)))
        print("EP:%d Model Saved on:" % epoch, os.path.join(self.model_path, '%s_ep%d.pt' % (file_name, epoch)))
        self.net.to(self.device)
