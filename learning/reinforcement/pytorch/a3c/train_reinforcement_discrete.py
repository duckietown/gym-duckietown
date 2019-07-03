import argparse
import logging
import datetime
import os
import sys
import numpy as np
from time import time


sys.path.append(os.path.join(os.getcwd(), "gym_duckietown"))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "learning"))
sys.path.append(os.path.join(os.getcwd(), "/learning/utils_global"))

# Duckietown Specific
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.reinforcement.pytorch.a3c import CustomOptimizer
from learning.reinforcement.pytorch.utils import seed, Logger
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper_a6

# PyTorch
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _train(args):
    # Ensure that multiprocessing works properly without deadlock...
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')

    env = launch_env()
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    env = DtRewardWrapper2(env)
    env = DiscreteWrapper_a6(env)

    # Set seeds
    seed(args.seed)

    logger = Logger("models")
    ckpt_dir, ckpt_path, log_dir = logger.get_log_dirs()

    shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    shape_action_space = env.action_space.n  # 3

    print("Initializing Global Network")
    global_net = a3c.Net(channels=1, num_actions=shape_action_space)
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = CustomOptimizer.SharedAdam(global_net.parameters(), lr=args.learning_rate)
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}

    if args.model_file is not None:
        cwd = os.getcwd()
        filepath = os.path.join(cwd, args.model_dir, args.model_file)
        checkpoint = torch.load(filepath)
        global_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        info = checkpoint['info']
        print('Loaded model:', args.model_file)

    print("Instantiating %i workers" % args.num_workers)

    workers = [
        a3c.Worker(global_net, optimizer, args, info, identifier=i, logger=logger)
        for i in range(args.num_workers)]

    print("Start training...")

    interrupted = False

    for w in workers:
        w.daemon = True
        w.start()

    try:
        [w.join() for w in workers]
    except KeyboardInterrupt:
        [w.terminate() for w in workers]
        interrupted = True

    if not interrupted or args.save_on_interrupt:
        print("Finished training.")

        if args.save_models:

            path = os.path.join(ckpt_dir, 'model-final.pth')

            torch.save({
                'model_state_dict': global_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'info': info
            }, path)

            print("Saved model to:",  f"{ckpt_dir}/model-final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--action_update_steps", default=10, type=int)  # after how many steps the agent chooses a new action
    parser.add_argument("--max_steps", default=20_000_000, type=int)  # Max time steps to run environment for
    parser.add_argument("--max_episode_steps", default=1e4, type=int)  # Max steps per episode
    parser.add_argument("--steps_until_sync", default=20, type=int)  # Steps until nets are synced
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # Learning rate for the net
    parser.add_argument("--gamma", default=0.99, type=float)  # Discount factor
    parser.add_argument("--num_workers", default=4, type=int)  # Number of processes to spawn
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument("--save_frequency", default=1500, type=int)  # Whether or not models are saved
    parser.add_argument('--save_on_interrupt', default=True)
    parser.add_argument('--model_dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument('--model_file', type=str, default=None)  # Name of the model to load
    parser.add_argument('--graphical_output', default=False)  # Whether to render the observation in a window
    parser.add_argument('--env', default=None)
    _train(parser.parse_args())
