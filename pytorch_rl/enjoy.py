import argparse
import os
import types
import time

import numpy as np
import torch
from torch.autograd import Variable
from pytorch_rl.vec_env.dummy_vec_env import DummyVecEnv

from envs import make_env

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--start-container', action='store_true', default=False,
                    help='start the Duckietown container image')

args = parser.parse_args()

env = make_env(args.env_name, args.seed, 0, None, args.start_container)
env = DummyVecEnv([env])

actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)

def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

render_func('human')
obs = env.reset()
update_current_obs(obs)

window = env.envs[0].unwrapped.window
@window.event
def on_key_press(symbol, modifiers):
    from pyglet.window import key
    import sys
    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    return

try:
    while True:
        value, action, _, states = actor_critic.act(Variable(current_obs, volatile=True),
                                                    Variable(states, volatile=True),
                                                    Variable(masks, volatile=True),
                                                    deterministic=True)
        states = states.data
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        print(cpu_actions)
        #print(np.tanh(cpu_actions[0]))

        # Obser reward and next obs
        obs, reward, done, _ = env.step(cpu_actions)
        time.sleep(0.08)

        masks.fill_(0.0 if done else 1.0)

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs)

        render_func('human')

except:
    env.envs[0].unwrapped.close()
    time.sleep(0.25)
