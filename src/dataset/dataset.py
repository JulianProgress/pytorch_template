import os

import numpy as np
import torch
from torch.utils import data

# DataPath
DEFINED_DATA_PATH = ''  # Basic Path


def defined_dataset_loader(batch_size):
    train = np.load(os.path.join(DEFINED_DATA_PATH, 'train.npz'))
    test = np.load(os.path.join(DEFINED_DATA_PATH, 'test.npz'))
    valid = np.load(os.path.join(DEFINED_DATA_PATH, 'valid.npz'))

    train_X, train_y = train['X'], train['y']
    valid_X, valid_y = valid['X'], valid['y']
    test_X, test_y = test['X'], test['y']

    trainloader = get_loader(train_X.reshape(train_X.shape[0], -1), train_y, batch_size)
    validloader = get_loader(valid_X.reshape(valid_X.shape[0], -1), valid_y, batch_size)
    testloader = get_loader(test_X.reshape(test_X.shape[0], -1), test_y, batch_size)

    return trainloader, validloader, testloader


def get_loader(x, y, batch_size):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    #     print(x.shape, y.shape)
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size, shuffle=True)
    return loader


def data_preprocess(X, y):
    # TODO: make own preprocessing steps
    return X, y
