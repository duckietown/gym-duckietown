from .train import launch_env, teacher
from .learner import NeuralNetworkPolicy
from .model import Squeezenet
from .algorithms import DAgger
import argparse
import os

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '-i', default=10, type=int)
    parser.add_argument('--horizon', '-r', default=64, type=int)
    parser.add_argument('--num-outputs', '-n', default=2, type=int)
    parser.add_argument('--model-path', '-mp', default="", type=str)
    parser.add_argument('--map-name', '-m', default="loop_empty", type=str)
    return parser

if __name__ == '__main__':
    parser = process_args()
    input_shape = (120,160)
    max_velocity = 0.7

    config = parser.parse_args()
    # launching environment and testing on different maps using map randomization
    environment = launch_env(config.map_name, randomize_maps_on_reset=True)
    
    task_horizon = config.horizon
    task_episode = config.episode

    if not(os.path.isfile(config.model_path)):
        raise Exception('Model File not found')

    model = Squeezenet(num_outputs=config.num_outputs, max_velocity=max_velocity)

    learner = NeuralNetworkPolicy(
        model=model,
        optimizer= None,
        dataset=None,
        storage_location="",
        input_shape=input_shape,
        max_velocity = max_velocity,
        model_path = config.model_path
    )

    algorithm = DAgger(env=environment,
                        teacher=teacher(environment, max_velocity),
                        learner=learner,
                        horizon = task_horizon,
                        episodes=task_episode,
                        alpha = 0,
                        test = True)
    
    algorithm.train(debug=True)  #DEBUG to show simulation

    environment.close()



