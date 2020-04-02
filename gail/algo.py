
import numpy as np
import torch
import torch.optim as optim
import sys
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter

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

        self.writer = SummaryWriter(comment='---{}/{}'.format(args.env_name, args.training_name))

        if args.imitation:
            print("using imitate!")

            self.update_generator = self.imitate

        elif update_with == "PPO":
            print("using ppo!")
            self.update_generator = self.ppo
        elif update_with == "POLICY GRADIENT":
            self.update_generator = policy_gradient
        elif update_with == "TRPO":
            self.update_generator = trpo
        elif update_with == "BEHAVIOUR CLONE":
            self.update_generator = self.imitate
        else:
            self.update_generator = self.vanilla_loss

        


        self.loss_fn = nn.BCELoss()
        

    def get_expert_trajectories(self, episodes, steps, expert_agent):
        trajectories = {"observations": [],
                        "actions": [],
                        }
        for ep in range(episodes):
            while True:
                try:
                    o, a, l, r, m, v = [], [], [], [], [], []

                    obs = self.env.reset()
                    for step in range(0, steps):
                        o.append(torch.FloatTensor(obs))
                        action = expert_agent.predict(None)

                        a.append(torch.FloatTensor(action))
                        obs, reward, done, info = self.env.step(action)


                    trajectories["observations"].append(torch.stack(o))
                    trajectories["actions"].append(torch.stack(a))

                    break
                except ValueError or KeyboardInterrupt:
                    break
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    pass

        for key in trajectories:
            trajectories[key] = torch.cat(trajectories[key])

        self.expert_trajectories = trajectories
        return

    def get_policy_trajectory(self, episodes, steps):
        trajectories = {"observations": [],
                        "actions": [],
                        "log_probs": [],
                        "rewards": [],
                        "masks":[],
                        "values":[],
                        "next_value":0}

        for ep in range(episodes):
            # while True:
            #     try:
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
                l.append(dist.log_prob(action))
                v.append(value)

                action = action.squeeze().data.cpu().numpy()

                obs, reward, done, info = self.env.step(action)

                _, next_value = self.generator(torch.FloatTensor(obs).unsqueeze(0).to(device))

                mask = 0 if done or step == steps-1 else 1
                m.append(mask)

            trajectories["observations"].append(torch.stack(o))
            trajectories["actions"].append(torch.stack(a))
            trajectories["log_probs"].append(torch.stack(l))
            trajectories["rewards"].append(torch.stack(r))
            trajectories["masks"].append(torch.FloatTensor(m))
            trajectories["values"].append(torch.stack(v))
            trajectories["next_value"] = next_value
            break
                # except ValueError:
                #     break
                # except KeyboardInterrupt:
                #     break
                # except:
                #     print("Unexpected error:", sys.exc_info()[0])
                #     pass
        
        for key in trajectories:
            if key != "next_value":
                trajectories[key] = torch.cat(trajectories[key])
            # print(key, trajectories[key].shape)

        return trajectories

    def train(self, epochs):

        best_reward = 0
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
                loss_d = self.update_discriminator(obs_batch, act_batch, policy_action, epoch)
        
            #Update Generator
            loss_g = self.update_generator(obs_batch, act_batch, policy_action, epoch)

            print('epcoh %d, D loss=%.5f, G loss=%.5f' % (epoch, loss_d, loss_g))
            #Save Checkpoint
            if epoch % 20 == 0:
                reward = self.eval()
                if reward>best_reward:
                    best_reward = reward
                    torch.save(self.generator.state_dict(), '{}/g-{}'.format(self.args.env_name, self.args.training_name))
                    torch.save(self.discriminator.state_dict(), '{}/d-{}'.format(self.args.env_name, self.args.training_name))
            torch.cuda.empty_cache()

        return

    def update_discriminator(self, obs_batch, act_batch, model_actions, epoch):
        if self.args.imitation:
            return 0
        self.d_optimizer.zero_grad()
        prob_expert = self.discriminator(obs_batch,act_batch)
        prob_policy = self.discriminator(obs_batch, model_actions)

        # if self.update_loss == "BCE":     
        exp_label = torch.full((self.args.batch_size,1), 0, device=device).float()
        policy_label = torch.full((self.args.batch_size,1), 1, device=device).float()

        expert_loss = self.loss_fn(prob_expert, exp_label)
        policy_loss = self.loss_fn(prob_policy, policy_label)

        loss = expert_loss + policy_loss

        loss.backward(retain_graph=True)
        self.d_optimizer.step()
            
        # else:
        #     loss = -(prob_expert.mean() - prob_policy.mean())
        
        self.writer.add_scalar("discriminator/loss", loss.data, epoch)
        return loss.data
        

    def eval(self):

        policy_actions = self.generator.get_means(self.expert_trajectories['observations'].to(device))
        reward = abs(self.expert_trajectories['actions'].to(device) - policy_actions).sum()
        return reward

    def enjoy(self):
        self.generator.eval().to(device)

        obs = self.env.reset()

        # max_count = 0
        while True:
            obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            action = self.generator.get_means(obs)

            action = action.squeeze().data.cpu().numpy()
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
        
        trajectories = self.get_policy_trajectory(steps=self.args.steps, episodes=1)
        observations, actions, log_probs, rewards, masks, values, next_value = trajectories['observations'],\
                                                                                trajectories['actions'],\
                                                                                trajectories['log_probs'],\
                                                                                trajectories['rewards'],\
                                                                                trajectories['masks'],\
                                                                                trajectories['values'],\
                                                                                trajectories['next_value']

        returns = compute_gae(next_value, rewards, masks, values, self.args)

        returns   = torch.cat(returns).detach().view(len(returns),1).to(device)
        log_probs = log_probs.detach().to(device).detach()
        values    = values.detach().to(device).detach()

        states    = observations.to(device)
        actions   = actions.to(device)
        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        loss = self.do_ppo_step(states, actions, log_probs, returns, advantage, self.args, self.generator, self.g_optimizer, epoch)

        return loss

    def vanilla_loss(self, obs_batch, act_batch, policy_action, epoch):
        self.g_optimizer.zero_grad()
        loss = self.discriminator(obs_batch,policy_action).log().mean()
        loss.backward()
        self.g_optimizer.step()
        self.writer.add_scalar("generator/loss", loss.data, epoch)

        return loss.data

    def policy_gradient(self, obs_batch, act_batch, epoch):
        policy_trajectories = get_policy_trajectory(1, 50)

        self.g_optimizer.zero_grad()
        loss = self.discriminator(obs_batch,act_batch).log().mean()
        loss.backward()
        self.g_optimizer.step()
        self.writer.add_scalar("generator/loss", loss.data, epoch)
        return loss.data

    def imitate(self, obs_batch, act_batch, policy_action, epoch):
        self.g_optimizer.zero_grad()
        loss = (policy_action - act_batch).norm(2).mean()

        loss.backward()
        
        self.g_optimizer.step()
        return loss.data

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

        for e in range(args.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = G(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action.squeeze(1)).unsqueeze(1)

                ratio = (new_log_probs.to(device) - old_log_probs.to(device)).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * advantage

                # print("logprobs", new_log_probs)
                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                # print("actorloss", actor_loss.data, "criticloss",critic_loss.data, "entropy",entropy)

                # print(args.critic_discount, critic_loss, actor_loss, args.entropy_beta, entropy)
                loss = args.critic_discount* critic_loss + actor_loss - args.entropy_beta * entropy
                # print(loss)
                G_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                G.float()
                G_optimizer.step()

                for p in G.parameters():
                    p = torch.clamp(p, -0.01,0.01)
                            
                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy
                
                count_steps += 1
        self.writer.add_scalar("returns", (sum_returns / count_steps).data, epoch)
        self.writer.add_scalar("advantage", (sum_advantage / count_steps).data, epoch)
        self.writer.add_scalar("loss_actor", (sum_loss_actor / count_steps).data, epoch)
        self.writer.add_scalar("loss_critic", (sum_loss_critic / count_steps).data, epoch)
        self.writer.add_scalar("entropy", (sum_entropy / count_steps).data, epoch)
        self.writer.add_scalar("loss_total", (sum_loss_total / count_steps).data, epoch)

        return (sum_loss_total / count_steps).data

def trpo():
    pass


def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.shape[0]
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // 5):
        rand_ids = np.random.randint(0, batch_size, 5)
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


if __name__ == "__main__":
    pass