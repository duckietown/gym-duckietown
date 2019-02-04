#!/usr/bin/env python3

import argparse
import sys
import numpy as np

from utils.teacher import PurePursuitExpert
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

from imitation.tensorflow.model import TensorflowModel

def _train(args):
    print("Running Expert for {} Episodes of {} Steps".format(args.episodes, args.steps))
    print("Training Learning for {} Epochs with Batch Size of {}".format(args.epochs, args.batch_size))

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []

    # let's collect our samples
    for episode in range(0, args.episodes):
        print("Starting episode", episode)
        for steps in range(0, args.steps):
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            
        env.reset()

    env.close()

    actions = np.array(actions)
    observations = np.array(observations)

    model = TensorflowModel(
        observation_shape=observation_shape,  # from the logs we've got
        action_shape=action_shape,  # same
        graph_location=args.model_directory,  # where do we want to store our trained models
        seed=args.seed  # to seed all random operations in the model (e.g., dropout)
    )

    for i in range(args.epochs):
        # we defined the batch size, this can be adjusted according to your computing resources
        loss = None
        for batch in range(0, len(observations), args.batch_size):
            print("Training batch", batch)
            loss = model.train(
                observations=observations[batch:batch + args.batch_size],
                actions=actions[batch:batch + args.batch_size]
            )

        # every 10 epochs, we store the model we have
        if i % 10 == 0:
            model.commit()

    print("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=3, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=50, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="imitation/tensorflow/models/", type=str, help="Where to save models")

    args = parser.parse_args()

    _train(args)