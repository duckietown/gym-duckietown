import argparse
import logging
import datetime
import os

# Duckietown Specific
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.reinforcement.pytorch.a3c import CustomOptimizer
from learning.reinforcement.pytorch.utils import seed
from learning.utils.wrappers import *

# PyTorch
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _train(args):
    env = gym.make(args.env).unwrapped

    # Set seeds
    seed(args.seed)

    print("Initializing Global Network")
    # Global Network
    global_net = a3c.Net(channels=1, num_actions=env.action_space.n)  # global net that's updated by the workers
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = CustomOptimizer.SharedAdam(global_net.parameters(), lr=args.learning_rate)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}

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

    if not interrupted:
        print("Finished training.")

        if args.save_models:
            cwd = os.getcwd()
            filedir = args.model_dir

            try:
                os.makedirs(os.path.join(cwd, filedir))
            except FileExistsError:
                # directory already exists
                pass

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + 'a3c-disc-pong.pth'
            path = os.path.join(cwd, filedir, filename)
            torch.save(global_net.state_dict(), path)
            print("Saved model to:", path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_episodes", default=4000, type=int)  # Max time steps to run environment for
    parser.add_argument("--steps_until_sync", default=20, type=int)  # Max time steps to run environment for
    parser.add_argument("--learning_rate", default=0.0001, type=float)  # Learning rate for the net
    parser.add_argument("--gamma", default=0.99, type=float)  # Learning rate for the net
    parser.add_argument("--tau", default=0.99, type=float)  # generalized advantage estimation discount
    parser.add_argument("--num_workers", default=2, type=int)  # Number of processes to spawn
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument('--model-dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument('--graphical_output', default=False)  # Whether to render the observation in a window
    parser.add_argument('--env', default='PongDeterministic-v0')
    _train(parser.parse_args())
