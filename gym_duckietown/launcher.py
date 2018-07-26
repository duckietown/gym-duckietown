from gym_duckietown.config import DEFAULTS
from duckietown_slimremote.networking import make_pull_socket, has_pull_message, receive_data, make_pub_socket, \
    send_gym
import os

from gym_duckietown.envs import DuckietownEnv

DEBUG = False


def main():
    """ Main launcher that starts the gym thread when the command 'duckietown-start-gym' is invoked
    """

    # get parameters from environment (set during docker launch) otherwise take default
    map = os.getenv('DUCKIETOWN_MAP', DEFAULTS["map"])
    domain_rand = bool(os.getenv('DUCKIETOWN_DOMAIN_RAND', DEFAULTS["domain_rand"]))
    max_steps = os.getenv('DUCKIETOWN_MAX_STEPS', DEFAULTS["max_steps"])

    # if a challenge is set, it overrides the map selection

    challenge = os.getenv('DUCKIETOWN_CHALLENGE', "")
    if challenge in ["LF", "LFV"]:
        print("Launching challenge:", challenge)
        map = DEFAULTS["challenges"][challenge]

    print("Using map:", map)

    env = DuckietownEnv(
        map_name=map,
        # draw_curve = args.draw_curve,
        # draw_bbox = args.draw_bbox,
        max_steps=max_steps,
        domain_rand=domain_rand
    )
    obs = env.reset()

    publisher_socket = None
    command_socket, command_poll = make_pull_socket()

    print("Simulator listening to incoming connections...")

    while True:
        if has_pull_message(command_socket, command_poll):
            success, data = receive_data(command_socket)
            if not success:
                print(data)  # in error case, this will contain the err msg
                continue

            reward = 0  # in case it's just a ping, not a motor command, we are sending a 0 reward
            done = False  # same thing... just for gym-compatibility

            if data["topic"] == 0:
                obs, reward, done, misc = env.step(data["msg"])
                if DEBUG:
                    print('step_count = %s, reward=%.3f, done = %s' % (env.unwrapped.step_count, reward, done))
                if done:
                    env.reset()

            if data["topic"] == 1:
                print("received ping:", data)

            if data["topic"] == 2:
                obs = env.reset()

            # can only initialize socket after first listener is connected - weird ZMQ bug
            if publisher_socket is None:
                publisher_socket = make_pub_socket(for_images=True)

            if data["topic"] in [0, 1]:
                send_gym(publisher_socket, obs, reward, done)
