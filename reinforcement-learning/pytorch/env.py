import gym
import gym_duckietown

def launch_env(id=None):
    env = None
    if id is None:
        # Needed for Running on Sagemaker.
        import os
        #status = os.system("Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &")
        os.environ['DISPLAY'] = ":99"
        
        # Launch the environment
        from gym_duckietown.simulator import Simulator
        env = Simulator(
            seed=123, # random seed
            map_name="loop_empty",
            max_steps=500001, # we don't want the gym to reset itself
            domain_rand=0,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4, # start close to straight
            full_transparency=True,
            distortion=True,
        )
    else:
        env = gym.make(id)

    return env
