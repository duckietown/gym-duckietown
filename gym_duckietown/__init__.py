from gym.envs.registration import register


register(
    id='Duckie-SimpleSim-v0',
    entry_point='gym_duckietown.envs:SimpleSimEnv',
    reward_threshold=900.0
)

register(
    id='Duckietown-v0',
    entry_point='gym_duckietown.envs:DuckietownEnv',
    reward_threshold=900.0
)

register(
    id='Duckiebot-v0',
    entry_point='gym_duckietown.envs:DuckiebotEnv',
    reward_threshold=900.0
)
