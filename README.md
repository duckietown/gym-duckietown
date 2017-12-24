# Gym-Duckietown

Duckietown environment for OpenAI gym. Connects to a remote ROS simulation
using ZeroMQ.

Installation
------------

Requirements:
- Python 3
- OpenAI gym
- numpy
- scipy
- pyglet
- rospy

First, install rospy:

```
sudo apt-get install python-rospy
```

Clone the repository and install the other dependencies with `pip3`:

```python3
git clone https://github.com/duckietown/simulator.git
cd gym
pip3 install -e .
```

Reinforcement learning code forked from [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
is included under [/basicrl](/basicrl). If you wish to use this code, you
can install its dependencies as follows:

```
cd basicrl

# PyTorch
conda install pytorch torchvision -c soumith

# Dependencies needed by OpenAI baselines
sudo apt install libopenmpi-dev zlib1g-dev cmake

# OpenAI baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip3 install -e .
cd ..

# Other requirements
pip3 install -r requirements.txt
```

Usage
-----

To run the standalone UI application, which allows you to control the robot manually:

```python3
./standalone.py
```

The standalone application will launch the gym environment, receive
camera images received and send actions (keyboard commands) back.

To train a reinforcement learning agent, you can use the code provided under [/basicrl](/basicrl). I recommend using the A2C or ACKTR implementations.
A sample command to launch training is:

```
python3 basicrl/main.py --no-vis --env-name Duckie-SimpleSim-Discrete-v0 --num-processes 1 --num-stack 4 --num-steps 20 --algo a2c --num-frames 1000000 --lr 0.0002 --entropy-coef 0.01 --max-grad-norm 0.5
```

Then, to visualize the result of training, you can run the following command.
Note that you can do this while the training process is still ongoing.

```
python3 basicrl/enjoy.py --env-name Duckie-SimpleSim-Discrete-v0 --num-stack 4 --load-dir trained_models/a2c
```

Reinforcement Learning Notes
----------------------------

Reinforcement learning algorithms are extremely sensitive to hyperparameters. Choosing the
wrong set of parameters could prevent convergence completely, or lead to unstable performance over
training. You will likely want to experiment. A learning rate that is too low can lead to no
learning happening. A learning rate that is too high can lead to an unstable or suboptimal
fixed-point.

The reward values are currently rescaled into the [0,1] range, because the RL code in
`basicrl` doesn't do reward clipping, and deals poorly with large reward values. Also
note that changing the reward function might mean you also have to retune your choice
of hyperparameters.
