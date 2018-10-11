from functools import reduce
import operator

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

class GradReverse(torch.autograd.Function):
    """
    Gradient reversal layer
    """

    def __init__(self, lambd=1):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def print_model_info(model):
    modelSize = 0
    for p in model.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(model))
    print('Total model size: %d' % modelSize)

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

def gen_batch(gen_data_fn, batch_size=2):
    """
    Returns a tuple of PyTorch Variable objects
    gen_data is expected to produce a tuple
    """

    assert batch_size > 0

    data = []
    for i in range(0, batch_size):
        data.append(gen_data_fn())

    # Create arrays of data elements for each variable
    num_vars = len(data[0])
    arrays = []
    for idx in range(0, num_vars):
        vals = []
        for datum in data:
            vals.append(datum[idx])
        arrays.append(vals)

    # Make a variable out of each element array
    vars = []
    for array in arrays:
        var = make_var(np.stack(array))
        vars.append(var)

    return tuple(vars)

from .utils_images import *
