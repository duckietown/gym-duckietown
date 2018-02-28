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

        self.conv1 = nn.Conv2d(3, 32, 6, stride=4)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)

        self.linear1 = nn.Linear(32 * 6 * 6, 256)
        self.linear2 = nn.Linear(256, 1)

        self.apply(initWeights)

    def forward(self, image):
        batch_size = image.size(0)

        x = image

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)

        #print(x.size())
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)

        return x

    def printInfo(self):
        modelSize = 0
        for p in self.parameters():
            pSize = reduce(operator.mul, p.size(), 1)
            modelSize += pSize
        print(str(self))
        print('Total model size: %d' % modelSize)

def genData():
    image = env.reset().copy()
    image = image.transpose(2, 0, 1)

    dist, dotDir, angle = env.getLanePos()
    targets = np.array([angle])

    return image, targets

def genBatch(batch_size=4):
    images = []
    targets = []

    for i in range(0, batch_size):
        img, out = genData()
        images.append(img)
        targets.append(out)

    assert len(images) == batch_size
    assert len(images) == len(targets)

    images = np.stack(images)
    targets = np.stack(targets)
    assert images.shape[0] == batch_size
    assert targets.shape[0] == batch_size

    return images, targets

def train(model, optimizer, image, target):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    output = model(image)

    loss = (output - target).norm(2).mean()
    loss.backward()
    optimizer.step()

    error = (output - target).abs().mean()

    return loss.data[0], error.data[0]

env = SimpleSimEnv()
env.reset()
obs_space = env.observation_space

model = Model(obs_space)
model.printInfo()
model.cuda()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

avg_error = 0

for epoch in range(1, 1000000):
    startTime = time.time()
    images, targets = genBatch()
    images = Variable(torch.from_numpy(images).float()).cuda()
    targets = Variable(torch.from_numpy(targets).float()).cuda()
    genTime = int(1000 * (time.time() - startTime))

    startTime = time.time()
    loss, error = train(model, optimizer, images, targets)
    trainTime = int(1000 * (time.time() - startTime))

    avg_error = avg_error * 0.995 + 0.005 * error

    print('gen time: %d ms' % genTime)
    print('train time: %d ms' % trainTime)
    print('epoch %d, loss=%.3f, error=%.3f' % (epoch, loss, avg_error))
