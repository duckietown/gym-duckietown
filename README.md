# Gym-Duckietown

Duckietown environment for OpenAI gym. Connects to a remote ROS simulation
using ZeroMQ.

Installation
------------

Requirements:
- Python 2
- OpenAI gym
- numpy
- scipy
- pyglet

Clone the repository and install the other dependencies with `pip2`:

```python2
git clone https://github.com/duckietown/simulator.git
cd gym
pip2 install -e .
```

To run the standalone UI application:

```python2
./standalone.py
```

The standalone application will connect to the ROS bridge node, display
camera images received and send actions (keyboard commands) back. By
default, the ROS bridge is assumed to be running on localhost.
