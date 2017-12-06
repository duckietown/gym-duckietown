from gym.envs.registration import register

register(
    id='Duckietown-v0',
    entry_point='gym_duckietown.envs:DuckietownEnv',
    # Threshold at which the environment is considered successfully solved
    reward_threshold=1000.0,
)

register(
    id='Duckietown-Discrete-v0',
    entry_point='gym_duckietown.envs:DiscreteEnv',
    # Threshold at which the environment is considered successfully solved
    reward_threshold=1000.0,
)
