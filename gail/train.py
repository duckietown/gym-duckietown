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
    # env = ActionWrapper(env)
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
        
    env.close()

    actions = np.array(actions)
    observations = np.array(observations)
    torch.save(actions, 'data_a.pt')
    torch.save(observations, 'data_o.pt')


def _train(args):
    
    writer = SummaryWriter(comment="generator_only")
    
        
    # writer.add_histogram("expert angles", actions[:,1])
    # writer.add_histogram("expert velocities", actions[:,0])

    data = ExpertTrajDataset(args)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = Generator(action_dim=2)
    model.train().to(device)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    avg_loss = 0
    
    model_actions = []

    for i_batch, sample_batched in enumerate(dataloader):
        epoch = i_batch
        optimizer.zero_grad()
        obs_batch = sample_batched['observation'].float().to(device).reshape((args.batch_size*100, 3, 160,120))
        act_batch= sample_batched['action'].float().to(device).reshape((args.batch_size*100, 2))
        
        model_actions = model(obs_batch)
        # writer.add_histogram("model velocities", model_actions[:,0])
        # writer.add_histogram("model angles", model_actions[:,1])
        loss = (model_actions - act_batch).norm(2).mean()
        loss.backward()
        optimizer.step()
        
        loss = loss.data.item()
        avg_loss = avg_loss * 0.995 + loss * 0.005        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        
        if epoch % 200 == 0:
            torch.save(model.state_dict(), 'test.pt')
            print('epoch %d, loss=%.3f' % (epoch, avg_loss))
            break
#             writer.add_graph(Generator, obs_batch)
        # if avg_loss <= 0.1 and epoch > 200:
        #     torch.save(model.state_dict(), 'test.pt')
        #     print('epoch %d, loss=%.3f' % (epoch, avg_loss))
        #     break


    writer.close()

#     for epoch in range(args.epochs):
        
#         optimizer.zero_grad()

#         batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
#         obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
#         act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

#         print(obs_batch.size())
#         model_actions = model(obs_batch)
#         writer.add_histogram("model velocities", model_actions[:,0])
#         writer.add_histogram("model angles", model_actions[:,1])
#         loss = (model_actions - act_batch).norm(2).mean()
#         loss.backward()
#         optimizer.step()
        
#         loss = loss.data.item()
#         avg_loss = avg_loss * 0.995 + loss * 0.005        
#         writer.add_scalar('Loss/train', avg_loss, epoch)
        
        
#         if epoch % 200 == 0:
#             torch.save(model.state_dict(), 'test.pt')
#             print('epoch %d, loss=%.3f' % (epoch, avg_loss))
# #             writer.add_graph(Generator, obs_batch)
    
#     writer.close()

def _trainGAIL(args):
    
    writer = SummaryWriter(comment="gail")
    
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

    avg_g_loss = 0
    avg_d_loss = 0
    for epoch in range(args.epochs):
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

        ## Sample trajectories
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)
        print(obs_batch.size())

        G_act_batch = G(obs_batch)
        
        D_loss = torch.log(D(obs_batch,act_batch)) + torch.log(1-D(obs_batch,G_act_batch))
        D_loss = D_loss.mean()
       
        avg_d_loss = avg_d_loss * 0.995 + D_loss * 0.005

        writer.add_scalar('Discriminator/loss', avg_d_loss, epoch)

        D_loss.backward(retain_graph=True)
        D_optimizer.step()
        

        G_loss = torch.log(1-D(obs_batch,G_act_batch)).mean()
#         G_loss = -torch.log(D(obs_batch,G_act_batch)).mean() #TF generator loss
        G_loss.backward()
        G_optimizer.step
        
        avg_g_loss = avg_g_loss * 0.995 + G_loss * 0.005

        
        writer.add_scalar('Generator/loss', avg_g_loss, epoch)

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(G.state_dict(), 'G_model.pt')
            torch.save(D.state_dict(), 'D_model.pt')

            print('epoch %d, loss=%.3f' % (epoch, avg_g_loss))
        
        torch.cuda.empty_cache()
