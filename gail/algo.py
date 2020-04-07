
import numpy as np
import torch
import torch.optim as optim
import sys
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
from gail.models import *

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class GAIL_Agent():

    def __init__(self, env, args, 
                generator:nn.Module, 
                update_with:str,
                discriminator=None, 
                g_optimizer= None,
                d_optimizer=None, 
                stochastic=True):

        self.env = env
        self.expert_trajectories = None
        self.training_name = args.training_name

        self.stochastic = stochastic
        
        self.generator = generator
        self.discriminator = discriminator

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.args = args


        if self.args.train:
            self.writer = SummaryWriter(comment='---{}/{}'.format(args.env_name, args.training_name))
        
        if args.imitation:
            print("using imitate!")
            self.update_generator = self.imitate
        elif update_with == "PPO":
            print("using ppo!")
            self.update_generator = self.ppo
        elif update_with == "POLICY GRADIENT":
            self.update_generator = self.policy_gradient
        elif update_with == "TRPO":
            self.update_generator = trpo
        elif update_with == "BEHAVIOUR CLONE":
            self.update_generator = self.imitate
        else:
            print("using discrim loss")
            self.update_generator = self.vanilla_loss

        

        self.validation_set = {}

        self.loss_fn = nn.BCELoss()
        

    def get_expert_trajectories(self, episodes, steps, expert_agent):
        trajectories = {"observations": [],
                        "actions": [],
                        "rewards":[],
                        }
        for e in range(episodes):
            if self.args.env_name == "duckietown":
                while True:
                    try:
                        o, a, l, r, m, v = [], [], [], [], [], []

                        obs = self.env.reset()
                        for step in range(0, steps):
                            o.append(torch.FloatTensor(obs))
                            action = expert_agent.predict(obs)

                            a.append(torch.FloatTensor(action))
                            obs, reward, done, info = self.env.step(action)
                            r.append(torch.FloatTensor([reward]))
                      
                        trajectories["observations"].append(torch.stack(o))
                        trajectories["actions"].append(torch.stack(a))
                        
                        trajectories["rewards"].append(torch.stack(r))

                        break
                    except ValueError or KeyboardInterrupt:
                        break
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        pass
            elif self.args.env_name in  ['CartPole-v1']:
                o, a, l, r, m, v = [], [], [], [], [], []

                obs = self.env.reset()
                for step in range(0, steps):
                    o.append(torch.FloatTensor(obs))
                    action = expert_agent.predict(obs)

                    obs, reward, done, info = self.env.step(action)
                    r.append(torch.FloatTensor([reward]))


                    # self.env.render()


                    if type(action) != list:
                        action = [action]
                    a.append(torch.FloatTensor(action))
                    if done:
                        break                    
                trajectories["observations"].append(torch.stack(o))
                trajectories["actions"].append(torch.stack(a))
                trajectories["rewards"].append(torch.stack(r))
        for key in trajectories:
            trajectories[key] = torch.cat(trajectories[key])

        self.expert_trajectories = trajectories
        return trajectories

    def get_policy_trajectory(self, episodes, steps):
        trajectories = {"observations": [],
                        "actions": [],
                        "log_probs": [],
                        "rewards": [],
                        "masks":[],
                        "values":[],
                        "next_value":0}
        for ep in range(episodes):
            if self.args.env_name == "CartPole-v1":
                o, a, l, r, m, v = [], [], [], [], [], []
                obs = self.env.reset()
                for step in range(0, steps):
                    obs = torch.FloatTensor(obs)
                    o.append(obs)

                    obs = obs.unsqueeze(0).to(device)
                    action = self.generator.sample_action(obs)
                    a.append(action)

                    action = action.to(device)

                    r.append(self.discriminator(obs,action)) 

                    dist, value = self.generator(obs)
                    l.append(log_prob(dist,action))
                    v.append(value)

                    action = action.squeeze().data.cpu()
                    
                    action = torch.clamp(action,0,1)
                    action = int(action)
                    # try:
                    obs, reward, done, info = self.env.step(action)
                    # except AssertionError:
                    #     print("Unexpected error:", sys.exc_info()[0])
                    #     print(action)

                    # if self.args.env_name == "CartPole-v1":
                    #     self.env.render()
                    _, next_value = self.generator(torch.FloatTensor(obs).unsqueeze(0).to(device))
                    mask = 0 if done or step == steps-1 else 1
                    m.append(mask)
                    if done:
                        break


                trajectories["observations"].append(torch.stack(o))
                trajectories["actions"].append(torch.stack(a))
                trajectories["log_probs"].append(torch.stack(l))
                trajectories["rewards"].append(torch.stack(r))
                trajectories["masks"].append(torch.FloatTensor(m))
                trajectories["values"].append(torch.stack(v))
                trajectories["next_value"] = next_value
            else:
                while True:
                    try:
                        o, a, l, r, m, v = [], [], [], [], [], []
                        obs = self.env.reset()
                        for step in range(0, steps):
                            obs = torch.FloatTensor(obs)
                            o.append(obs)

                            obs = obs.unsqueeze(0).to(device)
                            action = self.generator.sample_action(obs)
                            a.append(action)

                            action = action.to(device)

                            r.append(self.discriminator(obs,action)) 

                            dist, value = self.generator(obs)
                            l.append(log_prob(dist, action))
                            v.append(value)

                            action = action.squeeze().data.cpu().numpy()

                            if self.args.env_name == "CartPole-v1":
                                action = int(action)

                            obs, reward, done, info = self.env.step(action)
                            if self.args.env_name == "CartPole-v1":
                                self.env.render()
                            _, next_value = self.generator(torch.FloatTensor(obs).unsqueeze(0).to(device))
                            mask = 0 if done or step == steps-1 else 1
                            m.append(mask)
                            if done:
                                break


                        trajectories["observations"].append(torch.stack(o))
                        trajectories["actions"].append(torch.stack(a))
                        trajectories["log_probs"].append(torch.stack(l))
                        trajectories["rewards"].append(torch.stack(r))
                        trajectories["masks"].append(torch.FloatTensor(m))
                        trajectories["values"].append(torch.stack(v))
                        trajectories["next_value"] = next_value
                        break
                    except ValueError:
                        break
                    except KeyboardInterrupt:
                        break
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        pass
        
        for key in trajectories:
            if key != "next_value":
                trajectories[key] = torch.cat(trajectories[key])
            # print(key, trajectories[key].shape)

        return trajectories

    def train(self, epochs):

        if self.args.pretrain_D:
            for i in range(self.args.pretrain_D):
                batch_indices = np.random.randint(0, self.expert_trajectories['observations'].shape[0], (self.args.batch_size))
                obs_batch = self.expert_trajectories['observations'][batch_indices].float().to(device).data
                act_batch = self.expert_trajectories['actions'][batch_indices].float().to(device).data
                policy_action = self.generator.sample_action(obs_batch)
                self.update_discriminator(obs_batch, act_batch, policy_action, -i)
        best_reward = -float("inf")
        for epoch in range(epochs):
            batch_indices = np.random.randint(0, self.expert_trajectories['observations'].shape[0], (self.args.batch_size))
            obs_batch = self.expert_trajectories['observations'][batch_indices].float().to(device).data
            act_batch = self.expert_trajectories['actions'][batch_indices].float().to(device).data

            if self.stochastic:
                policy_action = self.generator.sample_action(obs_batch)
            else:
                policy_action = self.generator.get_means(obs_batch)
        
            #Update Discriminator
            if epoch % self.args.d_schedule == 0:
                for i in range(self.args.D_iter):
                    loss_d = self.update_discriminator(obs_batch, act_batch, policy_action, epoch)
        
            #Update Generator
            self.learning_rate = self.args.lrG * (1 - (epoch/self.args.epochs))
            loss_g = self.update_generator(obs_batch, act_batch, policy_action, epoch)

            print('epcoh %d, D loss=%.5f, G loss=%.5f' % (epoch, loss_d, loss_g))
            #Save Checkpoint
            reward = self.eval()
            print(reward.data)

            if epoch % 20 == 0:
                # loss_eval = self.eval()
                if reward>best_reward:
                    best_reward = reward
                    print("Made new checkpoint for model with loss of ", reward)
                    torch.save(self.generator.state_dict(), '{}/g-{}'.format(self.args.env_name, self.args.training_name))
                    torch.save(self.discriminator.state_dict(), '{}/d-{}'.format(self.args.env_name, self.args.training_name))
            torch.cuda.empty_cache()

        return

    def update_discriminator(self, obs_batch, act_batch, model_actions, epoch):
        if self.args.imitation:
            return torch.zeros(1)
        self.d_optimizer.zero_grad()
        prob_expert = self.discriminator(obs_batch,act_batch)
        prob_policy = self.discriminator(obs_batch, model_actions)

        if self.args.update_d == "BCE":     
            exp_label = torch.full((self.args.batch_size,1), 1, device=device).float()
            policy_label = torch.full((self.args.batch_size,1), 0, device=device).float()

            expert_loss = self.loss_fn(prob_expert, exp_label)
            policy_loss = self.loss_fn(prob_policy, policy_label)

            loss =  policy_loss + expert_loss

        elif self.args.update_d == "WGAN":
            loss = (prob_expert.mean() - prob_policy.mean()).norm(2)

        loss.backward(retain_graph=True)
        self.d_optimizer.step()

        for p in self.discriminator.parameters():
                p.data.clamp_(-0.01,0.01)

        self.writer.add_scalar("discriminator/loss", loss.data, epoch)
        return loss.data
        

    def eval(self):
        if self.args.env_name == "duckietown":
            policy_actions = self.generator.get_means(self.validation_set['observations'].to(device))
            reward = -abs(self.validation_set['actions'].to(device) - policy_actions).sum()
        
        else:
            trajectories = self.get_policy_trajectory(1, 100)
            reward = trajectories["rewards"].sum()
        # policy_actions = self.generator.get_means(self.expert_trajectories['observations'].to(device))
        # reward = abs(self.expert_trajectories['actions'].to(device) - policy_actions).sum()
        

        return reward

    def enjoy(self):
        self.generator.eval().to(device)
        obs = self.env.reset()

        # max_count = 0
        while True:
            obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            action = self.generator.get_means(obs)

            action = action.squeeze().data.cpu().numpy()
            if self.args.env_name == "CartPole-v1":
                action = int(action)
            # print("\nAction taken::", action, "\n")
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            

            # if max_count > 50:
            #     max_count = 0
            #     obs = env.reset()

            if done:
                if reward < 0:
                    print('*** FAILED ***')
                    time.sleep(0.7)
                # max_count += 1
                obs = self.env.reset()
                self.env.render()
                # if max_count > 10:
                #     break

    def ppo(self, obs_batch, act_batch, policy_action, epoch):
        
        trajectories = self.get_policy_trajectory(steps=self.args.ppo_steps, episodes=self.args.sampling_eps)
        observations, actions, log_probs, rewards, masks, values, next_value = trajectories['observations'],\
                                                                                trajectories['actions'],\
                                                                                trajectories['log_probs'],\
                                                                                trajectories['rewards'],\
                                                                                trajectories['masks'],\
                                                                                trajectories['values'],\
                                                                                trajectories['next_value']

        
        self.clip_param = self.args.clip_param * (1 - (epoch/self.args.epochs))

        returns = compute_gae(next_value, rewards, masks, values, self.args)
        returns   = torch.cat(returns).detach().view(len(returns),1).to(device)
        log_probs = log_probs.detach().to(device).detach()
        values    = values.detach().to(device).detach()

        states    = observations.to(device)
        actions   = actions.to(device)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        
        loss = self.do_ppo_step(states, actions, log_probs, returns, advantage, self.args, self.generator, self.g_optimizer, epoch)

        return loss
    
    def do_ppo_step(self, states, actions, log_probs, returns, advantages, args, G, G_optimizer, epoch):
        '''
        from https://github.com/colinskow/move37/blob/f57afca9d15ce0233b27b2b0d6508b99b46d4c7f/ppo/ppo_train.py#L63
        '''
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        last_dist = 0

        for pg in self.g_optimizer.param_groups:
            pg['lr'] = self.learning_rate

        buffer_size = states.size(0)
        shuffled_inds = np.random.permutation(buffer_size)
        batch_size = 32
        for e in range(args.ppo_epochs):
            # for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            for i in range(int(buffer_size/batch_size)):
                rand_ints = shuffled_inds[int(i*batch_size):int((i+1)*batch_size)]
                dist, value = G(states[rand_ints,:])

                # print("ahhh",dist.scale)
                entropy = dist.entropy().mean()
                # entropy[torch.isnan(entropy)] = 1e-9
                new_log_probs = log_prob(dist, actions[rand_ints,:].squeeze(1)).unsqueeze(1)
                # new_log_probs[torch.isnan(new_log_probs)] = 1e-9
                self.writer.add_scalar("PPO_STEP/entropy", entropy.data, epoch*int(buffer_size/batch_size)+count_steps)
                self.writer.add_scalar("PPO_STEP/new_log_probs", new_log_probs.mean().data, epoch*int(buffer_size/batch_size)+count_steps)
                self.writer.add_scalar("PPO_STEP/dist_means", dist.loc.mean().data, epoch*int(buffer_size/batch_size)+count_steps)
                self.writer.add_scalar("PPO_STEP/dist_scales", dist.scale.mean().data, epoch*int(buffer_size/batch_size)+count_steps)

                ratio = (new_log_probs.to(device) - log_probs[rand_ints,:].to(device)).exp()
                surr1 = ratio * advantages[rand_ints,:]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[rand_ints,:]
                # print(surr1.mean(), surr2.mean())
                # print("logprobs", new_log_probs)
                actor_loss  = -torch.min(surr1.mean(), surr2.mean())

                # print(actor_loss)
                critic_loss = (returns[rand_ints,:] - value).pow(2).mean()
                # print("actorloss", actor_loss.data, "criticloss",critic_loss.data, "entropy",entropy)

                # print(args.critic_discount, critic_loss, actor_loss, args.entropy_beta, entropy)
                loss = actor_loss - args.critic_discount*critic_loss +  args.entropy_beta*entropy
                # print(loss)
                G_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                G.float()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.25)
                G_optimizer.step()

  


                # for p in G.parameters():
                #     if torch.isnan(p.min()):
                #         print(p)
                #     p = torch.clamp(p, -0.01,0.01)
                            
                sum_returns += returns[rand_ints,:].mean()
                sum_advantage += advantages[rand_ints,:].mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy

                count_steps += 1
                last_dist = dist
                
        self.writer.add_scalar("PPO/returns", (sum_returns / count_steps).data, epoch)
        self.writer.add_scalar("PPO/advantage", (sum_advantage / count_steps).data, epoch)
        self.writer.add_scalar("PPO/loss_actor", (sum_loss_actor / count_steps).data, epoch)
        self.writer.add_scalar("PPO/loss_critic", (sum_loss_critic / count_steps).data, epoch)
        self.writer.add_scalar("PPO/entropy", (sum_entropy / count_steps).data, epoch)
        self.writer.add_scalar("PPO/loss_total", (sum_loss_total / count_steps).data, epoch)
        
        for param_group in self.g_optimizer.param_groups:
            self.writer.add_scalar("PPO/learning rate", param_group['lr'] , epoch)
            break
        return (sum_loss_total / count_steps).data

    def vanilla_loss(self, obs_batch, act_batch, policy_action, epoch):
        self.g_optimizer.zero_grad()
        loss = -self.discriminator(obs_batch,policy_action).log().mean()
        loss.backward()
        self.g_optimizer.step()
        self.writer.add_scalar("generator/loss", loss.data, epoch)

        return loss.data

    def policy_gradient(self, obs_batch, act_batch, policy_action, epoch):
        policy_trajectories = self.get_policy_trajectory(5, 50).to(device)

        self.g_optimizer.zero_grad()
        loss = policy_trajectories["rewards"].log().mean()
        loss.backward()
        self.g_optimizer.step()
        self.writer.add_scalar("generator/loss", loss.data, epoch)
        return loss.data

    def imitate(self, obs_batch, act_batch, policy_action, epoch):
        self.g_optimizer.zero_grad()
        for pg in self.g_optimizer.param_groups:
            pg['lr'] = self.learning_rate
        loss = (policy_action - act_batch).norm(2).mean()

        loss.backward()
        self.writer.add_scalar("imitation/loss", loss.data, epoch)
        self.g_optimizer.step()
        return loss.data

    
def trpo():
    pass


def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.shape[0]
    # generates random mini-batches until we have covered the full batch
    for _ in range(32):
        rand_ids = np.random.randint(0, batch_size, 1000)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        

def compute_gae(next_value, rewards, masks, values, args):
    values = torch.cat((values, next_value.unsqueeze(0)))
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + args.gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + (args.gamma * args.lam * masks[step] * gae)
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    
    return returns

def log_prob(dist, x):
    return ((1/(dist.scale*(2*math.pi)**(0.5)))*(math.e**(-0.5*((0-dist.loc)/dist.scale)**2)) + 1e-10).log()

if __name__ == "__main__":
    pass