# Gym-Duckietown

Duckietown simulator environment for OpenAI Gym.

Installation
------------

Requirements:
- Python 3
- OpenAI gym
- NumPy
- scipy
- OpenCV
- Pyglet
- PyTorch

Clone this repository and install the dependencies with `pip3`:

```python3
git clone https://github.com/duckietown/gym-duckietown.git
cd gym
pip3 install -e .
```

Reinforcement learning code forked from [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
is included under [/pytorch_rl](/pytorch_rl). If you wish to use this code, you
should install [PyTorch](http://pytorch.org/) as follows:

```
# PyTorch
conda install pytorch torchvision -c pytorch
```

Introduction
------------

This repository contains 3 different gym environments:
- `Duckie-SimpleSim-v0`
- `Duckietown-v0`
- `Duckiebot-v0`

The `Duckie-SimpleSim-v0` environment is a simple lane-following simulator
written in OpenGL (Pyglet). The `Duckietown-v0` environment is meant to
connect to a remote server running ROS/Gazebo which runs a more complete
simulation (but this simulation is often more buggy, slower, and trickier to
get working). The `Duckiebot-v0` is meant to connect to software running on
a real Duckiebot and remote control it. It is a tool to test that policies
trained in simulation can transfer to the real robot.

If you are interested in controlling your robot remotely with the
`Duckiebot-v0` environment, you will need to install the software
found in the [duck-remote-iface](https://github.com/maximecb/duck-remote-iface)
repository on your Duckiebot.

Usage
-----

To run the standalone UI application, which allows you to control the simulation or real
robot manually:

```python3
./standalone.py --env-name Duckie-SimpleSim-v0
```

The standalone application will launch the gym environment, receive
camera images received and send actions (keyboard commands) back.

To train a reinforcement learning agent, you can use the code provided under [/pytorch_rl](/pytorch_rl). I recommend using the A2C or ACKTR implementations.
A sample command to launch training is:

```
python3 pytorch_rl/main.py --no-vis --env-name Duckie-SimpleSim-Discrete-v0 --num-processes 1 --num-stack 1 --num-steps 20 --algo a2c --lr 0.0002 --max-grad-norm 0.5
```

Then, to visualize the result of training, you can run the following command.
Note that you can do this while the training process is still running.

```
python3 pytorch_rl/enjoy.py --env-name Duckie-SimpleSim-Discrete-v0 --num-stack 1 --load-dir trained_models/a2c
```

Reinforcement Learning Notes
----------------------------

Reinforcement learning algorithms are extremely sensitive to hyperparameters. Choosing the
wrong set of parameters could prevent convergence completely, or lead to unstable performance over
training. You will likely want to experiment. A learning rate that is too low can lead to no
learning happening. A learning rate that is too high can lead to an unstable or suboptimal
fixed-point.

The reward values are currently rescaled into the [0,1] range, because the RL code in
`pytorch_rl` doesn't do reward clipping, and deals poorly with large reward values. Also
note that changing the reward function might mean you also have to retune your choice
of hyperparameters.
