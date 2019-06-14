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

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=2,
                               padding=1)  # (16,30,40)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)  # (16,15,20)

        self.gru = nn.GRUCell(int(16 * (60 * .5 * .5) * (80 * .5 * .5)), 256)
        # self.dense_size = int(16 * (60 * .5 * .5) * (80 * .5 * .5))
        self.pi = nn.Linear(256, self.num_actions)  # actor
        self.v = nn.Linear(256, 1)  # critic

        # Weight & bias initialization
        for layer in [self.conv1, self.conv2, self.pi, self.v]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, inputs):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        gru_input = x.view(-1, int(16 * (60 * .5 * .5) * (80 * .5 * .5)))
        hx = self.gru(gru_input, hx)

        pi = self.pi(hx)
        values = self.v(hx)
        return pi, values, hx

    def choose_action(self, s):
        self.eval()
        s = v_wrap(s)
        s.unsqueeze_(0)
        logits, _, self.hx = self.forward((s, self.hx))
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values, self.hx = self.forward((s, self.hx))
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
        self.hx = None

    def run(self):
        import gym

        # We have to initialize the gym here, otherwise the multiprocessing will crash
        self.env = gym.make(self.env_name).unwrapped

        self.shape_obs_space = self.env.observation_space.shape
        self.shape_action_space = self.env.action_space.n

        self.local_net = Net(1, self.shape_action_space)  # local network

        done = True

        while self.global_episode.value < self.max_episodes:

            # Begin a new episode
            # sync global parameters
            self.local_net.load_state_dict(self.global_net.state_dict())

            obs = preprocess_state(self.env.reset())

            hx = torch.zeros(1, 256) if done else hx.detach()

            buffer_states = []
            buffer_values = []
            buffer_probs = []
            buffer_actions = []
            buffer_rewards = []

            t = 0
            episode_reward = 0

            done = False

            # Generate trajectory
            while done is False:
                t += 1
                self.total_step += 1

                if self.name == "Worker 0" and self.graphical_output:
                    self.env.render()

                # Run a forward pass through the local net
                state = v_wrap(obs)  # Ensure state is a pytorch tensor
                state.unsqueeze_(0)  # Add batch dimension
                logits, value, hx = self.local_net.forward((state, hx))

                # Sample an action
                action_probabilities = F.softmax(logits, dim=1).data  # Get probability for each action as tensor
                action = torch.distributions.Categorical(action_probabilities).sample().data[
                    0]  # Get action from distribution

                # Execute action
                new_obs, reward, done, info = self.env.step(action)
                new_obs = preprocess_state(new_obs)

                episode_reward += reward

                buffer_states.append(obs)
                buffer_actions.append(action)
                buffer_rewards.append(reward)
                buffer_values.append(value)
                buffer_probs.append(action_probabilities)

                obs = new_obs

                done = done or t >= self.max_steps_per_episode

                if done:
                    # Increase global episode counter
                    with self.global_episode.get_lock():
                        self.global_episode.value += 1

                    next_value = torch.zeros(1, 1) if done else self.local_net.forward((new_obs.unsqueeze(0), hx))[1]
                    buffer_values.append(next_value.detach())

                    # Calculate loss + gradients
                    loss = self.calc_loss(torch.cat(buffer_values), torch.cat(buffer_probs),
                                          torch.stack(buffer_actions), np.asarray(buffer_rewards))
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 40)

                    # Sync local net and global net
                    for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
                        if gp.grad is None:
                            gp._grad = lp.grad

                    # Backpropagation
                    self.optimizer.step()

                    # Update statistics
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

        self.env.close()
        self.res_queue.put(None)

    def calc_loss(self, values, probs, actions, rewards):
        np_values = values.view(-1).data.numpy()
        delta_t = np.asarray(rewards) + self.gamma * np_values[1:] - np_values[:-1]
        gather_indices = actions.clone().detach().view(-1, 1)
        logpys = probs.gather(1, gather_indices)
        gen_adv_est = discount(delta_t, self.gamma)
        policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

        # l2 loss over value estimator
        rewards[-1] += self.gamma * np_values[-1]
        discounted_r = discount(np.asarray(rewards), self.gamma)
        discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
        value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

        entropy_loss = -probs.sum()  # entropy definition, for entropy regularization
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def sync_nets(optimizer, local_net, global_net, done, next_state, state_buffer, action_buffer, reward_buffer, gamma):
    if done:
        v_next_state = 0.  # terminal
    else:
        flattened_input = v_wrap(next_state[None, :])
        v_next_state = local_net.forward((flattened_input, local_net.hx))[1].data.numpy()[0, 0]

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


def discount(x, gamma):
    from scipy.signal import lfilter
    return lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)
