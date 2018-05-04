# Gym-Duckietown

[Duckietown](http://duckietown.org/) self-driving car simulator environments for OpenAI Gym.

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_duckietown,
  author = {Maxime Chevalier-Boisvert, Florian Golemo, Yanjun Cao, Liam Paull},
  title = {Duckietown Environments for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/duckietown/gym-duckietown}},
}
```

This simulator was created as part of work done at the [MILA](https://mila.quebec/).

Introduction
------------

This repository contains 3 different gym environments:
- `SimpleSim-v0`
- `Duckietown-v0`
- `Duckiebot-v0`

The `SimpleSim-v0` environment is a simple lane-following simulator
written in OpenGL (Pyglet). It draws a loop of road with left and right turns,
along with obstacles in the background. It implements various forms of
[domain-randomization](https://blog.openai.com/spam-detection-in-the-physical-world/)
and basic differential-drive physics (without acceleration).

The `Duckietown-v0` environment is meant to connect to a [remote server running
ROS/Gazebo](https://github.com/duckietown/duckietown-sim-server) which runs a more
complete Duckietown simulation. This simulation is often more buggy, slower, and
trickier to get working, but it has a more accurate physics model and prettier
graphics, but no domain-randomization.

The `Duckiebot-v0` environment is meant to connect to software running on
a real Duckiebot and remotely control the robot. It is a tool to test that policies
trained in simulation can transfer to the real robot. If you want to
control your robot remotely with the `Duckiebot-v0` environment, you will need to
install the software found in the [duck-remote-iface](https://github.com/maximecb/duck-remote-iface)
repository on your Duckiebot.

If you simply want to experiment with lane-following, I would strongly
recommend that you start with the `SimpleSim-v0` environment, because
it is fast, relatively easy to install, and we know for a fact that
reinforcement learning policies can be successfully trained on it.

Installation
------------

Requirements:
- Python 3.5+
- OpenAI gym
- NumPy
- SciPy
- OpenCV
- Pyglet
- PyTorch

You can install all the dependencies, including PyTorch, using Conda as follows. If you are at MILA, this is the way to go:

```
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
conda env create -f environment.yaml
```

Alternatively, you can install all the dependencies except PyTorch with `pip3`:

```
git clone https://github.com/duckietown/gym-duckietown.git
cd gym-duckietown
pip3 install -e .
```

Reinforcement learning code forked from [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
is included under [/pytorch_rl](/pytorch_rl). If you wish to use this code, you
should install [PyTorch](http://pytorch.org/) as follows:

```
conda install pytorch torchvision -c pytorch
```

Usage
-----

To run the standalone UI application, which allows you to control the simulation or real
robot manually:

```
./standalone.py --env-name SimpleSim-v0
```

The `standalone.py` application will launch the Gym environment, display
camera images and send actions (keyboard commands) back to the simulator or robot.

To train a reinforcement learning agent, you can use the code provided under [/pytorch_rl](/pytorch_rl). I recommend using the A2C or ACKTR implementations.

A sample command to launch training is:

```
python3 pytorch_rl/main.py --no-vis --env-name Duckie-SimpleSim-Discrete-v0 --algo a2c --lr 0.0002 --max-grad-norm 0.5 --num-steps 20
```

Then, to visualize the results of training, you can run the following command. Note that you can do this while the training process is still running:

```
python3 pytorch_rl/enjoy.py --env-name Duckie-SimpleSim-Discrete-v0 --num-stack 1 --load-dir trained_models/a2c
```

Running Headless
----------------

The simulator uses the OpenGL API to produce graphics. This requires an X11 display to be running, which can be problematic if you are trying to run training code through on SSH, or on a cluster. There is a `headless.sh` script in this repository which will create a virtual display for the simulator.

The following illustrates how this can be used:

```
# Reserve a Debian 9 machine with 12GB ram, 2 cores and a GPU
# Note: this is specific to MILA users
sinter --reservation=res_stretch --mem=12000 -c2 --gres=gpu

# Activate the gym-duckietown Conda environment
source activate gym-duckietown

cd gym-duckietown

# Start a virtual display
source ./headless.sh

# You are now ready to train
```

Reinforcement Learning Notes
----------------------------

Reinforcement learning algorithms are extremely sensitive to hyperparameters. Choosing the
wrong set of parameters could prevent convergence completely, or lead to unstable performance over
training. You will likely want to experiment. A learning rate that is too low can lead to no
learning happening. A learning rate that is too high can lead unstable performance throughout
training or a suboptimal result.

The reward values are currently rescaled into the [0,1] range, because the RL code in
`pytorch_rl` doesn't do reward clipping, and deals poorly with large reward values. Also
note that changing the reward function might mean you also have to retune your choice
of hyperparameters.
