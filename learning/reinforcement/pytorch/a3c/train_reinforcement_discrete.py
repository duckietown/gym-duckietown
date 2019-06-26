import argparse
import logging
import datetime
import os
import sys
import numpy as np

# Duckietown Specific
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.reinforcement.pytorch.a3c import CustomOptimizer
from learning.reinforcement.pytorch.utils import seed
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, DiscreteWrapper

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
    env = DtRewardWrapper(env)
    env = DiscreteWrapper(env)

    # Set seeds
    seed(args.seed)

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
        a3c.Worker(global_net, optimizer, args, info, identifier=i)
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
            cwd = os.getcwd()
            filedir = args.model_dir

            try:
                os.makedirs(os.path.join(cwd, filedir))
            except FileExistsError:
                # directory already exists
                pass

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + 'a3c-disc-duckie.pth'
            path = os.path.join(cwd, filedir, filename)

            torch.save({
                'model_state_dict': global_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'info': info
            }, path)

            # torch.save(global_net.state_dict(), path)
            print("Saved model to:", path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_steps", default=40000, type=int)  # Max time steps to run environment for
    parser.add_argument("--steps_until_sync", default=20, type=int)  # Steps until nets are synced
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # Learning rate for the net
    parser.add_argument("--gamma", default=0.99, type=float)  # Discount factor
    parser.add_argument("--num_workers", default=3, type=int)  # Number of processes to spawn
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument('--model_dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument('--model_file', type=str, default='2019-06-26_18-06-53_a3c-disc-duckie.pth')  # Name of the model to load
    parser.add_argument('--graphical_output', default=False)  # Whether to render the observation in a window
    parser.add_argument('--env', default=None)
    parser.add_argument('--save_on_interrupt', default=True)
    _train(parser.parse_args())
