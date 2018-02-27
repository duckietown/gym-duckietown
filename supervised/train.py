#!/usr/bin/env python3

import time
from functools import reduce

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        #self.conv1_drop = torch.nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        #self.conv2_drop = torch.nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        #self.conv3_drop = torch.nn.Dropout2d(p=0.2)

        self.conv4 = nn.Conv2d(32, 32, 4, stride=2)

        #self.linear1_drop = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(32 * 5 * 5, 512)

        self.linear2 = nn.Linear(512, 1)

        # L1 loss, absolute value of element-wise difference
        self.lossFn = nn.L1Loss()

        """
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.4,
            nesterov=False
        )
        """

        self.optimizer = optim.Adadelta(
            self.parameters()
        )

        """
        self.optimizer = optim.RMSprop(
            self.parameters(),
            #lr=0.001,
        )
        """

    def forward(self, image):
        batch_size = image.size(0)

        x = self.conv1(image)
        #x = self.conv1_drop(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        #x = self.conv2_drop(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        #x = self.conv3_drop(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)

        # View the final convolution output as a flat vector
        x = x.view(-1, 32 * 5 * 5)
        #x = self.linear1_drop(x)
        x = self.linear1(x)
        x = F.leaky_relu(x)

        x = self.linear2(x)

        return x

    def train(self, image, target):
        """
        Expects image, string and labels to be in tensor form
        """

        image = Variable(torch.from_numpy(image).float())
        target = Variable(torch.from_numpy(target).float())

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        output = self(image)
        loss = self.lossFn(output, target)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().data[0]

env = SimpleSimEnv()
env.reset()
obs_space = env.observation_space

model = Model(obs_space)

def genData():
    image = env.reset().transpose(2, 0, 1)
    dist, dotDir, angle = env.getLanePos()
    output = np.array([dist])
    return image, output

def genBatch(batch_size = 32):
    images = []
    outputs = []

    for i in range(0, batch_size):
        img, out = genData()
        images.append(img)
        outputs.append(out)

    images = np.stack(images)
    outputs = np.stack(outputs)

    return images, outputs

avgLoss = None

for epoch in range(1, 100000):
    startTime = time.time()
    images, outputs = genBatch()
    genTime = int(1000 * (time.time() - startTime))

    startTime = time.time()
    loss = model.train(images, outputs)
    trainTime = int(1000 * (time.time() - startTime))

    if avgLoss is None:
        avgLoss = loss
    else:
        avgLoss = 0.99 * avgLoss + 0.01 * loss

    print('gen time: %d ms' % genTime)
    print('train time: %d ms' % trainTime)
    print('epoch %d, loss=%f' % (epoch, avgLoss))
    #print(loss)
