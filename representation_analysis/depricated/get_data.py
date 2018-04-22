import numpy as np
import torch
from torchvision import transforms

from pytorch_rl.arguments import get_args
from pytorch_rl.vec_env.dummy_vec_env import DummyVecEnv
from pytorch_rl.vec_env.subproc_vec_env import SubprocVecEnv
from pytorch_rl.envs import make_env

from PIL import Image

num_samples = 220000
args = get_args()

envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.start_container)
        for i in range(args.num_processes)]

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.num_processes > 1:
    envs = SubprocVecEnv(envs)
else:
    envs = DummyVecEnv(envs)

obs_shape = envs.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

current_obs = torch.zeros(args.num_processes, *obs_shape)

def update_current_obs(obs):
    shape_dim0 = envs.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

obs = envs.reset()
update_current_obs(obs)

for i in range(num_samples):
    obs = envs.reset()
    img = Image.fromarray((np.flipud(np.moveaxis(obs[0], [0, 1, 2], [2, 0, 1])) * 255).astype('uint8'))
    img.save(fp='representation_analysis/data/trajectories/{}.jpg'.format(i))
    if i+1 % 1000 == 0:
        print('got to sample {}'.format(i), flush=True)

print('done?')

