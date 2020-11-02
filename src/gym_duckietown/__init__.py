# coding=utf-8
__version__ = "6.0.11"

import logging

logging.basicConfig()
logger = logging.getLogger("gym-duckietown")
logger.setLevel(logging.INFO)

logger.info("gym-duckietown %s\n" % __version__)

import os

from gym.envs.registration import register

from .utils import get_subdir_path


def reg_map_env(map_file):
    _, map_name = os.path.split(map_file)
    map_name, _ = map_name.split(".")
    gym_id = "Duckietown-%s-v0" % map_name

    # logger.info('Registering gym environment id: %s' % gym_id)

    register(
        id=gym_id,
        entry_point="gym_duckietown.envs:DuckietownEnv",
        reward_threshold=400.0,
        kwargs={"map_name": map_file},
    )


# Register a gym environment for each map file available
for map_file in os.listdir(get_subdir_path("maps")):
    if "regress" not in map_file:
        reg_map_env(map_file)

register(id="MultiMap-v0", entry_point="gym_duckietown.envs:MultiMapEnv", reward_threshold=400.0)

register(id="Duckiebot-v0", entry_point="gym_duckietown.envs:DuckiebotEnv", reward_threshold=400.0)
