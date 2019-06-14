#!/usr/bin/env python
# manual

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np


def preprocess_state(obs):
    from scipy.misc import imresize
    return imresize(obs.mean(2), (60, 80)).astype(np.float32).reshape(1, 60, 80) / 255.


class Net(nn.Module):
    def __init__(self, channels, num_actions):
        super(Net, self).__init__()
        self.channels = channels
        self.num_actions = num_actions

        # Conv Filter output size:
        # o = output
        # p = padding
        # k = kernel_size
        # s = stride
        # d = dilation

        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=2, padding=1)  # (16,30,40)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)  # (16,15,20)

        self.dense_size = int(16 * (60 * .5 * .5) * (80 * .5 * .5))
        self.pi = nn.Linear(self.dense_size, self.num_actions)  # actor
        self.v = nn.Linear(self.dense_size, 1)  # critic

        # Weight & bias initialization
        for layer in [self.conv1, self.conv2, self.pi, self.v]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.contiguous().view(-1, self.dense_size)

        pi = self.pi(x)
        values = self.v(x)
        return pi, values

    def choose_action(self, s):
        self.eval()
        s = v_wrap(s)
        s.unsqueeze_(0)
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        critic_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (critic_loss + a_loss).mean()

        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, res_queue, name,
                 env_name='Breakout-v0', graphical_output=False, max_episodes=20, max_steps_per_episode=100,
                 sync_frequency=100, gamma=0.9):
        super(Worker, self).__init__()
        self.name = 'Worker %s' % name
        self.env_name = env_name
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
        self.env = gym.make(self.env_name).unwrapped

        self.shape_obs_space = self.env.observation_space.shape
        self.shape_action_space = self.env.action_space.n

        self.local_net = Net(1, self.shape_action_space)  # local network

        while self.global_episode.value < self.max_episodes:

            # sync global parameters
            self.local_net.load_state_dict(self.global_net.state_dict())

            obs = preprocess_state(self.env.reset())

            buffer_states = []
            buffer_actions = []
            buffer_rewards = []

            t = 0
            episode_reward = 0
            done = False

            while done is False:
                t += 1
                self.total_step += 1

                if self.name == "Worker 0" and self.graphical_output:
                    self.env.render()

                action = self.local_net.choose_action(obs)
                # print('Chosen action:', action)
                new_obs, reward, done, info = self.env.step(action)
                new_obs = preprocess_state(new_obs)

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
        v_wrap(np.array(state_buffer)),
        v_wrap(np.array(action_buffer), dtype=np.int64) if action_buffer[0].dtype == np.int64 else v_wrap(
            np.vstack(action_buffer)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients
    optimizer.zero_grad()
    loss.backward()

    # push local parameters to global
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    optimizer.step()


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)
