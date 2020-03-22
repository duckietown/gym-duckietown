from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from learning.utils.teacher import PurePursuitExpert
from gail.models import *

env = launch_env()
env = ResizeWrapper(env)
env = NormalizeWrapper(env) 
env = ImgWrapper(env)
# env = ActionWrapper(env)
env = DtRewardWrapper(env)
print("Initialized Wrappers")


observation_shape = (None, ) + env.observation_space.shape
action_shape = (None, ) + env.action_space.shape

model = Generator(action_dim=2)
model.load_state_dict(torch.load('test.pt'))
model.eval()
# model.to(device)
# Create an imperfect demonstrator

observations = []
actions = []

# let's collect our samples
def go(args):

    for episode in range(0, args.episodes):
        print("Starting episode", episode)
        observation = env.reset()
        for steps in range(0, args.steps):
            # use our 'expert' to predict the next action.
            observation = torch.tensor(observation, device="cpu").float()
            action = model.forward(observation.unsqueeze(0)).detach().numpy()[0]
            print(action)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            env.render()
        env.reset()
    env.close()