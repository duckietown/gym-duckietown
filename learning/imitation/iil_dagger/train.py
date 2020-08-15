import argparse
import os

from learning.imitation.iil_dagger.learning.iil_train_loop import train_iil
from learning.imitation.iil_dagger.utils.env import launch_env
from learning.imitation.iil_dagger.utils.save_manager import SaveManager
from .learner import NeuralNetworkPolicy
from .teacher import PurePursuitPolicy


def teacher(env, max_velocity):
    return PurePursuitPolicy(
        env=env,
        ref_velocity=max_velocity
    )

def legacy_parse(arg, arr):
    arg = str(arg)
    if "." in arg or "e" in arg:
        return float(arg)
    else:
        return arr[int(arg)]

def learning_rate_parse(arg):
    legacy_parse(arg, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

def decay_parse(arg):
    legacy_parse(arg, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])

# Best: Horizon64_LR1e-05_Decay0.95_BS32_epochs10_2020-06-0817:13:35.067047
def process_args():
    def str2bool(string):
        string = string.lower()

        yays = ["true", "1", "yes"]
        nays = ["false", "0", "no"]

        if string in yays:
            return True
        elif string in nays:
            return False
        else:
            raise Exception(f"Your provided argument, '{string}', is not {', or'.join(yays+nays)}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '-i', default=45, type=int)
    parser.add_argument('--horizon', '-r', default=64, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-3, type=learning_rate_parse)
    parser.add_argument('--decay', '-d', default=0.7, type=decay_parse)    # mixing decay
    parser.add_argument('--save-path', '-s', default='iil_baseline', type=str)
    parser.add_argument('--map-name', '-m', default="loop_empty", type=str)
    parser.add_argument('--num-outputs', '-n', default=2, type=int)

    parser.add_argument("--log_file", default="./log.txt", type=str)  # Maximum number of steps to keep in the replay buffer
    parser.add_argument("--max_velocity", default=0.7, type=float)
    parser.add_argument("--batch_size", default=32, type=int)

    return parser

if __name__ == '__main__':
    parser = process_args()
    input_shape = (120,160)
    epochs = 10

    config = parser.parse_args()

    if config.log_file != None:
        print('You asked for a log file. "Tee-ing" print to also print to file "' + config.log_file + '" now...')

        import subprocess, os, sys

        tee = subprocess.Popen(["tee", "./" + config.save_path + "/" + config.log_file], stdin=subprocess.PIPE)
        # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
        # of any child processes we spawn)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    # check for  storage path
    if not(os.path.isdir(config.save_path)):
        os.makedirs(config.save_path)
    # launching environment
    environment = launch_env(config.map_name, randomize_maps_on_reset=False)

    save_manager = SaveManager(config.save_path,
                               f"Horizon{config.horizon}_LR{config.learning_rate}_Decay{config.decay}_BS{config.batch_size}_epochs{epochs}_")
    task_horizon = config.horizon
    task_episode = config.episode

    iil_learner = NeuralNetworkPolicy(
        batch_size=config.batch_size,
        epochs=epochs,
        save_manager=save_manager,
        input_shape=input_shape,
        max_velocity = config.max_velocity,
        learning_rate=config.learning_rate,
        save_path=config.save_path,
    )

    train_iil(environment, iil_learner, task_horizon, task_episode, config.decay, max_velocity=0.7, _debug=False)

    print("Finished training. Closing the env now...")
    environment.close()
    print("Closed successfully.")
    exit()


