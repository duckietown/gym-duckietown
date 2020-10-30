import math
from gym_duckietown.envs import DuckietownEnv
import argparse

from .teacher import PurePursuitPolicy
from .learner import NeuralNetworkPolicy
from .model import Squeezenet
from .algorithms import DAgger
from .utils import MemoryMapDataset
import torch
import os


def launch_env(map_name, randomize_maps_on_reset=False, domain_rand=False):
    environment = DuckietownEnv(
        domain_rand=domain_rand, max_steps=math.inf, map_name=map_name, randomize_maps_on_reset=False
    )
    return environment


def teacher(env, max_velocity):
    return PurePursuitPolicy(env=env, ref_velocity=max_velocity)


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", "-i", default=10, type=int)
    parser.add_argument("--horizon", "-r", default=64, type=int)
    parser.add_argument("--learning-rate", "-l", default=2, type=int)
    parser.add_argument("--decay", "-d", default=2, type=int)
    parser.add_argument("--save-path", "-s", default="iil_baseline", type=str)
    parser.add_argument("--map-name", "-m", default="loop_empty", type=str)
    parser.add_argument("--num-outputs", "-n", default=2, type=int)
    return parser


if __name__ == "__main__":
    parser = process_args()
    input_shape = (120, 160)
    batch_size = 16
    epochs = 10
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # decays
    mixing_decays = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    # Max velocity
    max_velocity = 0.5

    config = parser.parse_args()
    # check for  storage path
    if not (os.path.isdir(config.save_path)):
        os.makedirs(config.save_path)
    # launching environment
    environment = launch_env(config.map_name)

    task_horizon = config.horizon
    task_episode = config.episode

    model = Squeezenet(num_outputs=config.num_outputs, max_velocity=max_velocity)
    policy_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[config.learning_rate])

    dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), config.save_path)
    learner = NeuralNetworkPolicy(
        model=model,
        optimizer=policy_optimizer,
        storage_location=config.save_path,
        batch_size=batch_size,
        epochs=epochs,
        input_shape=input_shape,
        max_velocity=max_velocity,
        dataset=dataset,
    )

    algorithm = DAgger(
        env=environment,
        teacher=teacher(environment, max_velocity),
        learner=learner,
        horizon=task_horizon,
        episodes=task_episode,
        alpha=mixing_decays[config.decay],
    )

    algorithm.train(debug=True)  # DEBUG to show simulation

    environment.close()
