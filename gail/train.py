import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.optim as optim

from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from learning.utils.teacher import PurePursuitExpert

from gail.models import *

from torch.utils.tensorboard import SummaryWriter
# from learning.imitation.pytorch.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    
    writer = SummaryWriter()
    
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []
    
    if args.get_samples:
    # let's collect our samples
        for episode in range(0, args.episodes):
            print("Starting episode", episode)
            for steps in range(0, args.steps):
                # use our 'expert' to predict the next action.
                action = expert.predict(None)
                observation, reward, done, info = env.step(action)
                observations.append(observation)
                actions.append(action)
            env.reset()
        env.close()
        
        actions = np.array(actions)
        observations = np.array(observations)
        torch.save(actions, 'data_a.pt')
        torch.save(observations, 'data_o.pt')

    else:
        actions = torch.load('data_a.pt')
        observations = torch.load('data_o.pt')


    
    model = Generator(action_dim=2)
    model.train().to(device)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.004,
        weight_decay=1e-3
    )

    avg_loss = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

        model_actions = model(obs_batch)
        
        loss = (model_actions - act_batch).norm(2).mean()
        loss.backward()
        optimizer.step()
        
        loss = loss.data.item()
        avg_loss = avg_loss * 0.995 + loss * 0.005

#         print('epoch %d, loss=%.3f' % (epoch, avg_loss))
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
#         writer.add_scalar('Loss/test', np.random.random(), n_iter)
#         writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#         writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(model.state_dict(), 'test.pt')
            print('epoch %d, loss=%.3f' % (epoch, avg_loss))


def _trainGAIL(args):
    
    writer = SummaryWriter()
    
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []
    
    if args.get_samples:
    # let's collect our samples
        for episode in range(0, args.episodes):
            print("Starting episode", episode)
            for steps in range(0, args.steps):
                # use our 'expert' to predict the next action.
                action = expert.predict(None)
                observation, reward, done, info = env.step(action)
                observations.append(observation)
                actions.append(action)
            env.reset()
        env.close()
        
        actions = np.array(actions)
        observations = np.array(observations)
        torch.save(actions, 'data_a.pt')
        torch.save(observations, 'data_o.pt')

    else:
        actions = torch.load('data_a.pt')
        observations = torch.load('data_o.pt')


    
    G = Generator(action_dim=2)
    G.train().to(device)

    D = Discriminator(action_dim=2)
    D.train().to(device)
    # weight_decay is L2 regularization, helps avoid overfitting
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=0.004,
        weight_decay=1e-3
    )

    D_optimizer = optim.Adam(
        D.parameters(),
        lr=0.004,
        weight_decay=1e-3
    )

    avg_loss = 0
    for epoch in range(args.epochs):
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

        ## Sample trajectories
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

        G_act_batch = G(obs_batch)
        

        loss = nn.NLLLoss()
        loss.backward()
        optimizer.step()
        
        loss = loss.data.item()
        avg_loss = avg_loss * 0.995 + loss * 0.005

#         print('epoch %d, loss=%.3f' % (epoch, avg_loss))
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
#         writer.add_scalar('Loss/test', np.random.random(), n_iter)
#         writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#         writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(model.state_dict(), 'test.pt')
            print('epoch %d, loss=%.3f' % (epoch, avg_loss))

