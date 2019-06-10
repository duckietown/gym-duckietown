#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, ActionClampWrapper, PreventBackwardsWrapper

"""
env = gym.make('Duckietown-udem1-v0')
SHAPE_OBS_SPACE = env.observation_space.shape[0]
SHAPE_ACTION_SPACE = env.action_space.shape[0]
GAMMA = 0.9
MAX_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 200
UPDATE_GLOBAL_NET_FREQUENCY = 5
"""
import numpy as np


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Net, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.input_size = state_shape[0] * state_shape[1] * state_shape[2]
        self.actor_1 = nn.Linear(self.input_size, 200)
        self.mu = nn.Linear(200, action_shape)
        self.sigma = nn.Linear(200, action_shape)
        self.critic_1 = nn.Linear(self.input_size, 100)
        self.v = nn.Linear(100, 1)

        for layer in [self.actor_1, self.mu, self.sigma, self.critic_1, self.v]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = x.contiguous().view(-1, self.input_size)
        a1 = F.relu6(self.actor_1(x))
        mu = 2 * torch.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001  # avoid 0
        c1 = F.relu6(self.critic_1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        s = v_wrap(s)
        mu, sigma, _ = self.forward(s)
        mu = mu.view(2, )
        sigma = sigma.view(2, )
        m = self.distribution(mu.view(2, ).data, sigma.view(2, ).data)
        actions = torch.clamp(m.sample(), -1, 1)
        return actions.numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        critic_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        actor_loss = -exp_v
        total_loss = (actor_loss + critic_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, res_queue, name,
                 graphical_output=False, max_episodes=20, max_steps_per_episode=100, sync_frequency=100,
                 gamma=0.9):
        super(Worker, self).__init__()
        self.name = 'Worker %i' % name
        self.env = None
        self.global_episode, self.global_episode_reward = global_episode, global_episode_reward
        self.res_queue = res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.total_step = 0
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_global_net_frequency = sync_frequency
        self.graphical_output = graphical_output
        self.gamma = gamma

    def run(self):
        import gym

        # We have to initialize the gym here, otherwise the multiprocessing will crash
        self.env = gym.make('Duckietown-udem1-v0')
        self.env = ResizeWrapper(self.env)
        self.env = NormalizeWrapper(self.env)
        self.env = ImgWrapper(self.env)  # to make the images from 160x120x3 into 3x160x120
        self.env = ActionWrapper(self.env)
        self.env = ActionClampWrapper(self.env)  # clip to range [-1, 1]
        self.env = PreventBackwardsWrapper(self.env)  # prevent duckiebot driving backwards..
        self.env = DtRewardWrapper(self.env)

        self.shape_obs_space = self.env.observation_space.shape  # (3, 120, 160)
        self.shape_action_space = self.env.action_space.shape[0]  # (2,)

        self.local_net = Net(self.shape_obs_space, self.shape_action_space)  # local network

        while self.global_episode.value < self.max_episodes:
            obs = self.env.reset()

            buffer_states = []
            buffer_actions = []
            buffer_rewards = []

            t = 0
            episode_reward = 0
            done = False

            while done is False:
                t += 1
                self.total_step += 1

                if self.graphical_output:
                    self.env.render()

                action = self.local_net.choose_action(obs)
                # print('Chosen action:', action)
                new_obs, reward, done, info = self.env.step(action)
                episode_reward += reward

                buffer_states.append(obs)
                buffer_actions.append(action)
                buffer_rewards.append(reward)

                if t == self.max_steps_per_episode:
                    with self.global_episode.get_lock():
                        self.global_episode.value += 1
                    done = True

                # Sync local net and global net
                if self.total_step % self.update_global_net_frequency == 0 or done:
                    # print(self.name + ': Syncing nets')

                    sync_nets(self.optimizer, self.local_net, self.global_net, done, new_obs, buffer_states,
                              buffer_actions, buffer_rewards, self.gamma)

                    buffer_states = []
                    buffer_actions = []
                    buffer_rewards = []

                    if done:
                        with self.global_episode.get_lock():
                            self.global_episode.value += 1

                        with self.global_episode_reward.get_lock():
                            # Moving average
                            if self.global_episode_reward.value == 0.:
                                self.global_episode_reward.value = episode_reward
                            else:
                                self.global_episode_reward.value = self.global_episode_reward.value * 0.9 + \
                                                                   episode_reward * 0.1
                        self.res_queue.put(self.global_episode_reward.value)

                        print(self.name, 'Global Episode:', self.global_episode.value,
                              '| Global Episode R: %.0f' % self.global_episode_reward.value)
                        break

                obs = new_obs
        self.env.close()
        self.res_queue.put(None)


def sync_nets(optimizer, local_net, global_net, done, next_state, state_buffer, action_buffer, reward_buffer, gamma):
    if done:
        v_next_state = 0.  # terminal
    else:
        flattened_input = v_wrap(next_state[None, :])
        v_next_state = local_net.forward(flattened_input)[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in reward_buffer[::-1]:  # reverse buffer r
        v_next_state = r + gamma * v_next_state
        buffer_v_target.append(v_next_state)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        v_wrap(np.vstack(state_buffer)),
        v_wrap(np.array(action_buffer), dtype=np.int64) if action_buffer[0].dtype == np.int64 else v_wrap(
            np.vstack(action_buffer)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # TODO: Do we need locks here?
    # calculate local gradients
    optimizer.zero_grad()
    loss.backward()

    # push local parameters to global
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    optimizer.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)
