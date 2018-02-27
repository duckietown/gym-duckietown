#!/usr/bin/env python3

import time
from functools import reduce
import operator

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def initWeights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Model(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        #self.batch_norm = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, 32, 8, stride=8)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        #self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        #self.conv4 = nn.Conv2d(32, 32, 4, stride=2)

        self.linear1 = nn.Linear(32 * 7 * 7, 1)

        #self.linear1 = nn.Linear(32 * 5 * 5, 1)
        #self.linear2 = nn.Linear(512, 256)
        #self.linear3 = nn.Linear(256, 1)

        #self.apply(initWeights)

    def forward(self, image):
        batch_size = image.size(0)

        # Note: this doesn't really seem to affect performance
        #x = self.batch_norm(image)
        x = image

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        #x = self.conv3(x)
        #x = F.relu(x)

        #x = self.conv4(x)
        #x = F.relu(x)

        # View the final convolution output as a flat vector
        #x = x.view(-1, 32 * 5 * 5)

        #print(x.size())
        x = x.view(-1, 32 * 7 * 7)

        x = self.linear1(x)
        #x = F.relu(x)
        #x = self.linear2(x)
        #x = F.relu(x)
        #x = self.linear3(x)

        return x

    def printInfo(self):
        modelSize = 0
        for p in self.parameters():
            pSize = reduce(operator.mul, p.size(), 1)
            modelSize += pSize
        print(str(self))
        print('Total model size: %d' % modelSize)

def genData():
    image = env.reset()
    image = image.transpose(2, 0, 1)
    #image = np.zeros_like(image)

    dist, dotDir, angle = env.getLanePos()
    targets = np.array([angle])

    #targets = np.array([env.curPos[0]])

    return image, targets

def genBatch(batch_size = 1):
    images = []
    targets = []

    for i in range(0, batch_size):
        img, out = genData()
        images.append(img)
        targets.append(out)

    assert len(images) == len(targets)

    images = np.stack(images)
    targets = np.stack(targets)

    assert images.shape[0] == batch_size
    assert targets.shape[0] == batch_size

    return images, targets

def train(model, lossFn, optimizer, image, target):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    output = model(image)

    loss = lossFn(output, target)
    loss.backward()
    optimizer.step()

    error = (output - target).abs().mean()

    return loss.data[0], error.data[0]

env = SimpleSimEnv()
env.reset()
obs_space = env.observation_space

model = Model(obs_space)
model.printInfo()
#model.cuda()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# L1 loss, absolute value of element-wise difference
#lossFn = nn.L1Loss()
lossFn = nn.SmoothL1Loss()














for epoch in range(1, 10000):

    startTime = time.time()
    images, targets = genBatch()
    images = Variable(torch.from_numpy(images).float())
    targets = Variable(torch.from_numpy(targets).float())
    genTime = int(1000 * (time.time() - startTime))

    startTime = time.time()
    loss, error = train(model, lossFn, optimizer, images, targets)
    trainTime = int(1000 * (time.time() - startTime))

    #print('gen time: %d ms' % genTime)
    #print('train time: %d ms' % trainTime)
    print('epoch %d, loss=%f, error=%f' % (epoch, loss, error))
    #print(loss)
