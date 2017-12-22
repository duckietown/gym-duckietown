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

Usage
-----

To run the standalone UI application, which allows you to control the robot manually:

```python3
./standalone.py
```

The standalone application will connect to the ROS bridge node, display
camera images received and send actions (keyboard commands) back. By
default, the ROS bridge is assumed to be running on localhost.

To train a reinforcement learning agent, you can clone the PyTorch code from [this repository](https://github.com/maximecb/pytorch-a2c-ppo-acktr). It has been modified and tested for
compatibility with `gym-duckietown`. I recommend using the PPO or ACKTR implementations.
A sample command to launch training is:

```
python3 main.py --env-name Duckietown-v0 --no-vis --num-processes 1 --algo acktr  --num-frames 10000000 --entropy-coef 0.22 --lr 0.0002
```

Then, to visualize the result of training, you can run the following command.
Note that you can do this while the training process is still ongoing.

```
python3 enjoy.py --env-name Duckie-SimpleSim-Discrete-v0 --num-stack 4 --load-dir trained_models/acktr
```
