# coding=utf-8
__version__ = "6.0.39"

from zuper_commons.logs import ZLogger

logger = ZLogger("gym-duckietown")

logger.debug(f"gym-duckietown version {__version__} path {__file__}\n")

import os

from gym.envs.registration import register

from .utils import get_subdir_path


def reg_map_env(map_file):
    _, map_name = os.path.split(map_file)
    map_name, _ = map_name.split(".")
    gym_id = f"Duckietown-{map_name}-v0"

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
