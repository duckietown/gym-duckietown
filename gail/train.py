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
from gail.dataloader import *

from torch.utils.tensorboard import SummaryWriter
# from learning.imitation.pytorch.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_expert_data(args):
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []
    
    # let's collect our samples
    for episode in range(0, args.episodes):
        print("Starting episode", episode)
        observations = []
        actions = []
        for steps in range(0, args.steps):
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            
        actions = np.array(actions)
        observations = np.array(observations)
        torch.save(actions, '{}/data_a_{}.pt'.format(args.data_directory,episode))
        torch.save(observations, '{}/data_o_{}.pt'.format(args.data_directory,episode))    
        env.reset()
    env.close()

    actions = np.array(actions)
    observations = np.array(observations)
    torch.save(actions, 'data_a.pt')
    torch.save(observations, 'data_o.pt')

def _train(args):
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    # env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    writer = SummaryWriter(comment="gail")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []

    # let's collect our samples
    if args.get_samples:
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
        torch.save(actions, '{}/data_a_.pt'.format(args.data_directory))
        torch.save(observations, '{}/data_o_.pt'.format(args.data_directory))    

    else:
        actions = torch.load('{}/data_a_.pt'.format(args.data_directory))
        observations = torch.load('{}/data_a_.pt'.format(args.data_directory))


    G = Generator(action_dim=2).to(device)
    D = Discriminator(action_dim=2).to(device)

    G_optimizer = optim.SGD(
        G.parameters(), 
        lr = 0.0004,
        weight_decay=1e-3
        )
    D_optimizer = optim.SGD(
        D.parameters(),
        lr = 0.0004,
    )


    avg_loss = 0
    avg_g_loss = 0
    loss_fn = nn.BCELoss()

    for epoch in range(args.epochs):
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

        model_actions = G(obs_batch)

        ## Update D

        exp_label = torch.full((args.batch_size,1), 1, device=device)
        policy_label = torch.full((args.batch_size,1), 0, device=device)

        D_optimizer.zero_grad()

        prob_expert = D(obs_batch,act_batch)
        loss = loss_fn(prob_expert, exp_label)

        prob_Generator = D(obs_batch,model_actions)
        loss += loss_fn(prob_Generator, policy_label)

        loss.backward(retain_graph=True)
        D_optimizer.step()

        ## Update G

        G_optimizer.zero_grad()

        loss_g = -D(obs_batch,model_actions)
        loss_g = loss_g.mean()
        loss_g.backward()
        G_optimizer.step()

        avg_loss = avg_loss * 0.995 + loss.item() * 0.005
        avg_g_loss = avg_g_loss * 0.995 + loss_g.item() * 0.005

        writer.add_scalar("D/loss", avg_loss, epoch) #should go towards 0.5
        writer.add_scalar("G/loss", avg_g_loss, epoch) #should go towards -inf?
        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(D.state_dict(), '{}/D.pt'.format(args.model_directory))
            torch.save(G.state_dict(), '{}/G.pt'.format(args.model_directory))

        torch.cuda.empty_cache()