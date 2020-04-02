from gail.algo import *
from gail.models import *

import argparse
import re

## Environment Variables
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=1, type=int, help="Train new model")
parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
parser.add_argument("--episodes", default=9, type=int, help="Number of epsiodes for experts")
parser.add_argument("--steps", default=50, type=int, help="Number of steps per episode")
parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
parser.add_argument("--epochs", default=201, type=int, help="Number of training epochs")
parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")
parser.add_argument("--data-directory", default="D:/Michael/Learning/duckietown_data/", type=str, help="Where to save generated expert data")
parser.add_argument("--lrG", default=0.0004, type=float, help="Generator learning rate")
parser.add_argument("--lrD", default=0.0004, type=float, help="Discriminator learning rate")
parser.add_argument("--get-samples", default=1, type=int, help="Generate expert data")
parser.add_argument("--use-checkpoint", default=0, type=int, help="Use checkpoint for training")
parser.add_argument("--checkpoint", default="", type=str, help="file name for checkpoint for training")
parser.add_argument("--training-name", default="last", type=str, help="file tag training type")
parser.add_argument("--rollout", default=1, type=int, help="file tag training type")
parser.add_argument("--d_schedule", default=5, type=int, help="number of times to train D vs G, negative means train D every G updates")
parser.add_argument("--pretrain", default=0, type=int, help="flag to run imitation learning instead")
parser.add_argument("--eval", default=1, type=int, help="flag to run eval script")
parser.add_argument("--eps", default=0.000001, type=float, help="epsilon for imitation learning")
parser.add_argument("--eval-steps", default=5, type=int, help="number of steps to evaluate policy on")
parser.add_argument("--eval-episodes", default=50, type=int)
parser.add_argument("--enjoy", default=0, type=int)
parser.add_argument("--pretrain-D", default=50, type=int)
parser.add_argument("--env_name", default="duckietown", type=str)

parser.add_argument("--gamma", default=0.995, type=float)
parser.add_argument("--lam", default=0.97, type=float)
parser.add_argument("--clip-param", default=0.2, type=float)
parser.add_argument("--entropy-beta", default=0.001, type=float)
parser.add_argument("--ppo-epsilon", default=0.02, type=float)
parser.add_argument("--ppo-epochs", default=6, type=int)
parser.add_argument("--ppo-steps", default=256, type=int)
parser.add_argument("--critic-discount", default=0.5, type=float)

parser.add_argument("--imitation", default=0, type=int)
parser.add_argument("--pretrain-name", default="imitate")


def main(args):
    ## Set cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    ## Setup environment

    # Duckietown environment
    from learning.utils.env import launch_env
    from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
        DtRewardWrapper, ActionWrapper, ResizeWrapper
    from learning.utils.teacher import PurePursuitExpert   

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    observation_shape = env.observation_space.shape
    action_dim = int(env.action_space.shape[0])

    # Duckietown expert
    expert = PurePursuitExpert(env=env)

    ## Setup Models


    print(observation_shape,action_dim)
    G = Generator(observation_shape, action_dim)
    G.to(device)
    D = Discriminator(observation_shape, action_dim)
    D.to(device)

    state_dict = torch.load('{}/g-{}'.format(args.env_name, args.pretrain_name), map_location=device)
    G.load_state_dict(state_dict)



    D_optimizer = optim.SGD(
        D.parameters(), 
        lr = args.lrD,
        weight_decay=1e-3
        )

    G_optimizer = optim.SGD(
        G.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )
    
    ## Setup GAIL

    gail_agent = GAIL_Agent(env, args, generator=G, discriminator=D, g_optimizer=G_optimizer,d_optimizer=D_optimizer,update_with="PPO")
    
    gail_agent.get_expert_trajectories(args.episodes, args.steps, expert)

    gail_agent.train(6000)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.enjoy:

        from learning.utils.env import launch_env
        from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
            DtRewardWrapper, ActionWrapper, ResizeWrapper
        from learning.utils.teacher import PurePursuitExpert   

        env = launch_env()
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env) 
        env = ImgWrapper(env)
        env = ActionWrapper(env)
        env = DtRewardWrapper(env)

        model = Generator(0,action_dim=2)
        state_dict = torch.load('{}/g-{}'.format(args.env_name, args.training_name), map_location=device)
        model.load_state_dict(state_dict)
        gail_agent = GAIL_Agent(env, args, model,"PPO")

        gail_agent.enjoy()

    main(args)



