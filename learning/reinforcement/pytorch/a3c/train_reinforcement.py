import argparse
import logging
import datetime
import os
import numpy as np

# Duckietown Specific
from learning.reinforcement.pytorch.a3c import a3c_continuous_simple as a3c
from learning.reinforcement.pytorch.a3c import CustomOptimizer
from learning.reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

# PyTorch
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _train(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    print("Initializing Global Network")
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    # Set seeds
    seed(args.seed)

    shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    shape_action_space = env.action_space.shape[0]  # (2,)

    print("Initialized Wrappers")

    # Global Network
    global_net = a3c.Net(shape_obs_space, shape_action_space)  # global net that's updated by the workers
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = CustomOptimizer.SharedAdam(global_net.parameters(), lr=args.learning_rate)
    global_episode, global_episode_reward, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    print("Instantiating %i workers" % args.num_workers)

    workers = [
        a3c.Worker(global_net, optimizer, global_episode, global_episode_reward, res_queue, name=i,
                   graphical_output=args.graphical_output, max_episodes=args.max_episodes,
                   max_steps_per_episode=args.max_steps_per_episode, sync_frequency=args.sync_frequency,
                   gamma=args.discount)
        for i in range(args.num_workers)]

    print("Start training...")

    [w.start() for w in workers]

    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]

    print("Finished training.")

    if args.save_models:
        cwd = os.getcwd()
        filedir = args.model_dir

        try:
            os.makedirs(os.path.join(cwd, filedir))
        except FileExistsError:
            # directory already exists
            pass

        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + 'a3c-cont.pth'
        path = os.path.join(cwd, filedir, filename)
        torch.save(global_net, path)
        print("Saved model to:", path)

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_episodes", default=100, type=int)  # Max time steps to run environment for
    parser.add_argument("--max_steps_per_episode", default=1000, type=int)  # Max time steps to run environment for
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--learning_rate", default=0.0002, type=float)  # Learning rate for the net
    parser.add_argument("--sync_frequency", default=128, type=int)  # Time steps until sync of the nets
    parser.add_argument("--num_workers", default=4, type=int)  # Batch size for both actor and critic
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument('--model-dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument("--graphical_output", default=True)  # Whether to render the observation in a window

    _train(parser.parse_args())
