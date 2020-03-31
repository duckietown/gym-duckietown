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

    G_optimizer = optim.Adam(
        G.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    V_optimizer = optim.SGD(
        V.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    avg_loss = 0
    avg_g_loss = 0
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()

    writer = SummaryWriter(comment='gail/{}'.format(args.training_name))
    if args.pretrain_D:
        for i in range(args.pretrain_D):
            batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
            obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device).data
            act_batch = torch.from_numpy(actions[batch_indices]).float().to(device).data
            
            model_actions = G.select_action(obs_batch, device=device)
            prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, loss_fn, D, obs_batch, act_batch, model_actions, writer, -1)

    for epoch in range(args.epochs):
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device).data
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device).data

        model_actions = G.select_action(obs_batch, device=device)

        ## Update D

        if args.D_train_eps > 0 and epoch % args.D_train_eps == 0:
            prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch)
        # elif args.D_train_eps>0:
        #     D_train_eps = args.D_train_eps
        #     prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, D_train_eps, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch)


        # Update G
        if True: 
            loss_g = do_policy_gradient(args, G, D, env, obs_batch, model_actions)
        # # if None:
        # loss_g = do_ppo_step(args, G, D, V, env, obs_batch, model_actions)
        
        loss_g.backward()
        G_optimizer.step()

        avg_g_loss = loss_g.item()
        avg_loss = loss.item() 

        writer.add_scalar("G/loss", avg_g_loss, epoch) #should go towards -inf?
        print('epoch %d, D loss=%.5f, G loss=%.5f' % (epoch, avg_loss, avg_g_loss))

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
    masks = []
    for episode in range(0, episodes):
        obs = env.reset()
        for step in range(0, steps):
            # use our 'expert' to predict the next action.
            obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            action = policy.select_action(obs, device=device)
            action = action.squeeze().data.cpu().numpy()
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            actions.append(action)

            mask = 0 if done or step == steps-1 else 1
            masks.append(mask)

            # env.render()
    return torch.FloatTensor(observations).to(device).data, torch.FloatTensor(actions).to(device).data, torch.FloatTensor(masks).to(device).data

def do_policy_gradient(args, G, D, env, obs_batch, model_actions):
    if args.rollout:
        obs, acts, masks = generate_trajectories(G, env, 10)
    else:
        obs = obs_batch
        acts = model_actions

    # loss_g = D(obs,acts).log().mean()
    loss_g = D(obs,acts).mean()

    return loss_g

def update_discriminator(args, D_optimizer, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch):
    D_optimizer.zero_grad()

    exp_label = torch.full((args.batch_size,1), 0, device=device).float()
    policy_label = torch.full((args.batch_size,1), 1, device=device).float()

    ##Making labels soft
    # exp_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1.1
    # exp_label = exp_label.clamp(0.7,1.2)

    # policy_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1
    # policy_label = policy_label.clamp(0,0.3)
    ##

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

    if epoch != -1:
        writer.add_scalar("expert D probability", torch.mean(prob_expert), epoch)
        writer.add_scalar("policy D probability", torch.mean(prob_policy), epoch)
        writer.add_scalar("D/loss", loss, epoch)

    return prob_expert.data, prob_policy.data, loss.data



def generalized_advantage_estimator(rewards, values, masks, args):
    
    returns = torch.Tensor(rewards.size(0),1)
    deltas = torch.Tensor(rewards.size(0),1)
    advantages = torch.Tensor(rewards.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]



    return advantages.to(device)
from torch.distributions import Distribution
def do_ppo_step(args, G, D, V, V_optimizer, env, obs_batch, model_actions):
    
    ## Rollout according to policy

    obs, acts, masks = generate_trajectories(G, env, args.steps)
    
    v_policy_old , s_policy_old = G.get_distributions(obs_batch)
    # ## Compute Value loss
    # V_optimizer.zero_grad()
    # values = V(obs)
    # rewards = -G(obs).log().mean().data
    # v_loss = (rewards-values).pow(2).mean()

    # ## Policy loss

    # advantage = generalized_advantage_estimator(rewards, values, masks, args)



    pass

def get_ppo_loss(old_dist, old_value, new_dist, new_value, actions, rewards, advantages, args):
    '''
    referenced from https://github.com/adik993/ppo-pytorch/blob/master/agents/ppo.py
    '''
    value_old_clipped = old_value + (new_value - old_value).clamp(-args.v_clip_range, args.v_clip_range)
    v_old_loss_clipped = (rewards - value_old_clipped).pow(2)
    v_loss = (rewards - new_value).pow(2)
    value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

    # Policy loss
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    advantage.detach_()
    log_prob = new_dist.log_prob(actions)
    log_prob_old = old_dist.log_prob(actions)
    ratio = (log_prob - log_prob_old).exp().view(-1)

    surrogate = advantage * ratio
    surrogate_clipped = advantage * ratio.clamp(1 - args.clip_range, 1 + args.clip_range)
    policy_loss = torch.min(surrogate, surrogate_clipped).mean()

    # Entropy
    entropy = new_dist.entropy().mean()

    # Total loss
    losses = policy_loss + args.c_entropy * entropy - args.c_value * value_loss
    total_loss = -losses
    return total_loss

    
    



    ## Compute Advantages

def get_kl(mu1,mu2,std1,std2):
    return  (std2/std1).log() + \
            ( (std1.pow(2)+ (mu1-mu2).pow(2)) / (2*std2.pow(2)) ) + \
            1/2
            


    