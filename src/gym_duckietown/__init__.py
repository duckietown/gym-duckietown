# coding=utf-8
__version__ = "6.1.0"

from zuper_commons.fs import AbsFilePath
from zuper_commons.logs import ZLogger

from duckietown_world.resources import list_maps2

logger = ZLogger("gym-duckietown")
import os

path = os.path.dirname(os.path.dirname(__file__))
logger.debug(f"gym-duckietown version {__version__} path {path}\n")

from gym.envs.registration import register

from .utils import get_subdir_path


def reg_map_env(map_name0: str, map_file: AbsFilePath):
    gym_id = f"Duckietown-{map_name0}-v0"

    # logger.info('Registering gym environment id: %s' % gym_id)

    register(
        id=gym_id,
        entry_point="gym_duckietown.envs:DuckietownEnv",
        reward_threshold=400.0,
        kwargs={"map_name": map_file},
    )


for map_name, filename in list_maps2().items():
    # Register a gym environment for each map file available
    if "regress" not in filename:
        reg_map_env(map_name, filename)

register(id="MultiMap-v0", entry_point="gym_duckietown.envs:MultiMapEnv", reward_threshold=400.0)

register(id="Duckiebot-v0", entry_point="gym_duckietown.envs:DuckiebotEnv", reward_threshold=400.0)
