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

from gail.models import *
from gail.dataloader import *

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    from learning.utils.env import launch_env
    from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
        DtRewardWrapper, ActionWrapper, ResizeWrapper
    from learning.utils.teacher import PurePursuitExpert

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    observations = []    
    actions = []

    expert = PurePursuitExpert(env=env)
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
        torch.save(actions, '{}/data_a_t.pt'.format(args.data_directory))
        torch.save(observations, '{}/data_o_t.pt'.format(args.data_directory))    

    else:
        observations = torch.load('{}/data_o_t.pt'.format(args.data_directory))
        actions = torch.load('{}/data_a_t.pt'.format(args.data_directory))

    actions = np.array(actions)
    observations = np.array(observations)
    
    G = Generator(action_dim=2).train().to(device)
    D = Discriminator(action_dim=2).train().to(device)
    V = Value(action_dim=2).train().to(device)

    # state_dict = torch.load('models/G_imitate_2.pt'.format(args.checkpoint), map_location=device)
    # G.load_state_dict(state_dict)
    if args.checkpoint:
        state_dict = torch.load('models/G_{}.pt'.format(args.checkpoint), map_location=device)
        G.load_state_dict(state_dict)
        # state_dict = torch.load('models/D_{}.pt'.format(args.checkpoint), map_location=device)
        # D.load_state_dict(state_dict)

    D_optimizer = optim.SGD(
        D.parameters(), 
        lr = args.lrD,
        weight_decay=1e-3
        )

    G_optimizer = optim.SGD(
        G.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    V_optimizaer = optim.SGD(
        V.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    avg_loss = 0
    avg_g_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()

    writer = SummaryWriter(comment='gail/{}'.format(args.training_name))

    for epoch in range(args.epochs):
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

        model_actions = G(obs_batch)

        ## Update D

        if args.D_train_eps < 0 and epoch % args.D_train_eps == 0:
            D_train_eps = 1
            prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, D_train_eps, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch)
        elif args.D_train_eps>0:
            D_train_eps = args.D_train_eps
            prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, D_train_eps, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch)


        # Update G
        if True: 
            loss_g = do_policy_gradient(args, G, D, env, obs_batch, model_actions)
        if None:
            loss_g = do_ppo_step(args)
        loss_g.backward()
        G_optimizer.step()

        avg_g_loss = loss_g.item()
        avg_loss = loss.item() 

        writer.add_scalar("G/loss", avg_g_loss, epoch) #should go towards -inf?
        print('epoch %d, D loss=%.3f, G loss=%.3f' % (epoch, avg_loss, avg_g_loss))

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(D.state_dict(), '{}/D_{}_epoch_{}.pt'.format(args.model_directory,args.training_name,epoch))
            torch.save(G.state_dict(), '{}/G_{}_epoch_{}.pt'.format(args.model_directory,args.training_name,epoch))
        torch.save(D.state_dict(), '{}/D_{}.pt'.format(args.model_directory,args.training_name))
        torch.save(G.state_dict(), '{}/G_{}.pt'.format(args.model_directory,args.training_name))
        torch.cuda.empty_cache()

    # writer.add_graph("generator", G)
    # writer.add_graph("discriminator",D)


def generate_trajectories(policy, env, steps, episodes=2):
    observations = []
    actions = []
    
    for episode in range(0, episodes):
        obs = env.reset()
        for steps in range(0, steps):
            # use our 'expert' to predict the next action.
            obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            action = policy(obs)
            action = action.squeeze().data.cpu().numpy()
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            # env.render()

    return torch.FloatTensor(observations).to(device), torch.FloatTensor(actions).to(device)

def do_policy_gradient(args, G, D, env, obs_batch, model_actions):
    if args.rollout:
        obs, acts = generate_trajectories(G, env, args.steps)
    else:
        obs = obs_batch
        acts = model_actions

    loss_g = D(obs,acts).log().mean()
    return loss_g

def update_discriminator(args, D_optimizer, D_train_eps, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch):
    D_optimizer.zero_grad()

    exp_label = torch.full((args.batch_size,1), 0, device=device).float()
    policy_label = torch.full((args.batch_size,1), 1, device=device).float()

    ##Making labels soft
    # exp_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1.1
    # exp_label = exp_label.clamp(0.7,1.2)

    # policy_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1
    # policy_label = policy_label.clamp(0,0.3)
    ##
    for _ in range(D_train_eps):
        prob_expert = D(obs_batch,act_batch)
        expert_loss = loss_fn(prob_expert, exp_label)
        # writer.add_scalar("expert D loss", expert_loss, epoch)

        prob_policy = D(obs_batch,model_actions)
        policy_loss = loss_fn(prob_policy, policy_label)
        # writer.add_scalar("policy D loss", policy_loss, epoch)

        loss = (expert_loss + policy_loss)

        # loss = -(prob_expert.mean() - prob_policy.mean())

        # if epoch % 10:
        loss.backward(retain_graph=True)
        D_optimizer.step()

        for p in D.parameters():
            p.data.clamp_(-0.01,0.01)
    writer.add_scalar("expert D probability", torch.mean(prob_expert), epoch)
    writer.add_scalar("policy D probability", torch.mean(prob_policy), epoch)
    writer.add_scalar("D/loss", loss, epoch)

    return prob_expert.data, prob_policy.data, loss.data


def generalized_advantage_estimator(rewards, values):
    pass
def do_ppo_step(args, G, D, V, env, obs_batch, model_actions):
    
    
    if args.rollout:
        obs, acts = generate_trajectories(G, env, args.steps)
    else:
        obs = obs_batch
        acts = model_actions


    