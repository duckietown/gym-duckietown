#!/usr/bin/env python3


import sys
import cv2
import numpy as np
from model import TensorflowModel
from teacher import PurePursuitExpert
from env import launch_env

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1234, type=int) # Sets Gym, TF, and Numpy seeds
parser.add_argument("--episodes", default=3, type=int) # Number of epsiodes for experts
parser.add_argument("--steps", default=50, type=int) # Number of steps per episode
parser.add_argument("--batch_size", default=32, type=int) # Training batch size
parser.add_argument("--epochs", default=1, type=int) # Number of training epochs

args = parser.parse_args()

# Hyperparameters
SEED = args.seed
EPISODES = args.episodes
STEPS = args.steps
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

print("Running Expert for {} Episodes of {} Steps".format(EPISODES, STEPS))
print("Training Learning for {} Epochs with Batch Size of {}".format(EPOCHS, BATCH_SIZE))

env = launch_env()

# this is an imperfect demonstrator... I'm sure you can construct a better one.
expert = PurePursuitExpert(env=env)

observations = []
actions = []

# let's collect our samples
for episode in range(0, EPISODES):
    print("Starting episode", episode)
    for steps in range(0, STEPS):
        # we use our 'expert' to predict the next action.
        action = expert.predict(None)
        observation, reward, done, info = env.step(action)
        # we can resize the image here
        observation = cv2.resize(observation, (80, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        observations.append(observation)
        actions.append(action)
        
    env.reset()

env.close()

# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 2)
STORAGE_LOCATION = '/opt/ml/model/'

actions = np.array(actions)
observations = np.array(observations)

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

# we trained for EPOCHS epochs
for i in range(EPOCHS):
    # we defined the batch size, this can be adjusted according to your computing resources...
    loss = None
    for batch in range(0, len(observations), BATCH_SIZE):
        print("Training batch", batch)
        loss = model.train(
            observations=observations[batch:batch + BATCH_SIZE],
            actions=actions[batch:batch + BATCH_SIZE]
        )

    # every 10 epochs, we store the model we have
    # but I'm sure that you're smarter than that, what if this model is worse than the one we had before
    if i % 10 == 0:
        model.commit()

print("Training complete!")
sys.exit(0)
