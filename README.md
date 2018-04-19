# Gym-Duckietown

[Duckietown](http://duckietown.mit.edu/) self-driving car simulator environment for OpenAI Gym.

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_minigrid,
  author = {Maxime Chevalier-Boisvert, Florian Golemo, Yanjun Cao, Liam Paull},
  title = {Duckietown Environments for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
}
```

Introduction
------------

This repository contains 3 different gym environments:
- `Duckie-SimpleSim-v0`
- `Duckietown-v0`
- `Duckiebot-v0`

The `Duckie-SimpleSim-v0` environment is a simple lane-following simulator
written in OpenGL (Pyglet). It draws a loop of road with left and right turns,
along with obstacles in the background. It implements various forms of
[domain-randomization](https://blog.openai.com/spam-detection-in-the-physical-world/)
and basic differential-drive physics (without acceleration).

The `Duckietown-v0` environment is meant to connect to a remote server runnin
ROS/Gazebo which runs a more complete Duckietown simulation. This simulation is
often more buggy, slower, and trickier to get working, but it has a more accurate
physics model and prettier graphics, but no domain-randomization.

The `Duckiebot-v0` environment is meant to connect to software running on
a real Duckiebot and remotely control the robot. It is a tool to test that policies
trained in simulation can transfer to the real robot. If you want to
control your robot remotely with the `Duckiebot-v0` environment, you will need to
install the software found in the [duck-remote-iface](https://github.com/maximecb/duck-remote-iface)
repository on your Duckiebot.

If you simply want to experiment with lane-following, I would strongly
recommend that you start with the `Duckie-SimpleSim-v0` environment, because
it is fast, relatively easy to install, and we know for a fact that
reinforcement learning policies can be successfully trained on it.

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
cd gym-duckietown
pip3 install -e .
```

Reinforcement learning code forked from [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
is included under [/pytorch_rl](/pytorch_rl). If you wish to use this code, you
should install [PyTorch](http://pytorch.org/) as follows:

```
# PyTorch
conda install pytorch torchvision -c pytorch
```

Usage
-----

To run the standalone UI application, which allows you to control the simulation or real
robot manually:

```python3
./standalone.py --env-name Duckie-SimpleSim-v0
```

The `standalone.py` application will launch the Gym environment, display
camera images and send actions (keyboard commands) back to the simulator or robot.

To train a reinforcement learning agent, you can use the code provided under [/pytorch_rl](/pytorch_rl). I recommend using the A2C or ACKTR implementations.

A sample command to launch training is:

```
python3 pytorch_rl/main.py --no-vis --env-name Duckie-SimpleSim-Discrete-v0 --num-processes 1 --num-stack 1 --num-steps 20 --algo a2c --lr 0.0002 --max-grad-norm 0.5
```

Then, to visualize the results of training, you can run the following command. Note that you can do this while the training process is still running:

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
