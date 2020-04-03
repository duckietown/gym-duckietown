from gail.algo import *
from gail.models import *

import argparse
import re

## Environment Variables
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=1, type=int, help="Train new model")
parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
parser.add_argument("--episodes", default=5, type=int, help="Number of epsiodes for experts")
parser.add_argument("--steps", default=50, type=int, help="Number of steps per episode")
parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
parser.add_argument("--epochs", default=201, type=int, help="Number of training epochs")
parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")
parser.add_argument("--data-directory", default="C:/Users/cps/Desktop/CSC2621/gym-duckietown/data", type=str, help="Where to save generated expert data")
parser.add_argument("--lrG", default=0.004, type=float, help="Generator learning rate")
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
parser.add_argument("--pretrain-D", default=10, type=int)
parser.add_argument("--env_name", default='CartPole-v1', type=str)

parser.add_argument("--eval-seed", default=0, type=int)


parser.add_argument("--gamma", default=0.995, type=float)
parser.add_argument("--lam", default=0.97, type=float)
parser.add_argument("--clip-param", default=0.2, type=float)
parser.add_argument("--entropy-beta", default=0.001, type=float)
parser.add_argument("--ppo-epsilon", default=0.02, type=float)
parser.add_argument("--ppo-epochs", default=6, type=int)
parser.add_argument("--ppo-steps", default=256, type=int)
parser.add_argument("--critic-discount", default=0.5, type=float)

parser.add_argument("--sampling-eps", default= 1, type=int)


parser.add_argument("--imitation", default=0, type=int)
parser.add_argument("--pretrain-name", default="imitate")
parser.add_argument("--update-d", default="WGAN")


def main(args):
    ## Set cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    ## Setup environment

    # Duckietown environment
    from learning.utils.env import launch_env
    from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
        DtRewardWrapper, ActionWrapper, ResizeWrapper, PixelWrapper
    from learning.utils.teacher import PurePursuitExpert, CartpoleController

    env = launch_env(args.env_name)
    if args.env_name == 'duckietown':
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env) 
        env = ImgWrapper(env)
        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        action_dim = 2
    else:
        # env = PixelWrapper(env)
        action_dim = 1

    observation_shape = env.observation_space.shape

    env.seed(args.seed)
    # Duckietown expert
    if args.env_name == "duckietown":
        expert = PurePursuitExpert(env=env)
    if args.env_name in ['CartPole-v1']:
        expert = CartpoleController(env=env)
    

    ## Setup Models


    print(observation_shape,action_dim)
    G = Generator(observation_shape, action_dim)
    G.to(device)
    D = Discriminator(observation_shape, action_dim)
    D.to(device)

    if args.use_checkpoint:
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

    
    if args.train:

        gail_agent = GAIL_Agent(env, args, generator=G, discriminator=D, g_optimizer=G_optimizer,d_optimizer=D_optimizer,update_with="PPO")

        env.seed(args.seed)
        gail_agent.get_expert_trajectories(args.episodes, args.steps, expert)
        gail_agent.train(args.epochs)
    
    if args.eval:
        G_random = Generator(observation_shape, action_dim)
        G_random.to(device)
        state_dict = torch.load('{}/g-{}'.format(args.env_name, args.training_name), map_location=device)
        G.load_state_dict(state_dict)

        gail_agent = GAIL_Agent(env, args, generator=G, discriminator=D, g_optimizer=G_optimizer,d_optimizer=D_optimizer,update_with="PPO")
        gail_agent_random = GAIL_Agent(env, args, generator=G_random, discriminator=D, g_optimizer=G_optimizer,d_optimizer=D_optimizer,update_with="PPO")

        gail_agent.env.seed(args.eval_seed)
        print(gail_agent.get_expert_trajectories(args.episodes, args.steps, expert)["rewards"].sum())

        gail_agent.env.seed(args.eval_seed)
        print(gail_agent.get_policy_trajectory(args.episodes, args.steps)["rewards"].sum())
        
        gail_agent_random.env.seed(args.eval_seed)
        print(gail_agent_random.get_policy_trajectory(args.episodes, args.steps)["rewards"].sum())



    env.close()


if __name__ == "__main__":
    args = parser.parse_args()

    if args.enjoy:

        from learning.utils.env import launch_env
        from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
            DtRewardWrapper, ActionWrapper, ResizeWrapper
        from learning.utils.teacher import PurePursuitExpert   
        env = launch_env(args.env_name)
        if args.env_name == 'duckietown':
            env = ResizeWrapper(env)
            env = NormalizeWrapper(env) 
            env = ImgWrapper(env)
            env = ActionWrapper(env)
            env = DtRewardWrapper(env)
            action_dim = 2
        else:
            action_dim = 1
        observation_shape = env.observation_space.shape


        model = Generator(observation_shape,action_dim=action_dim)
        state_dict = torch.load('{}/g-{}'.format(args.env_name, args.training_name), map_location=device)
        model.load_state_dict(state_dict)
        gail_agent = GAIL_Agent(env, args, model,"PPO")

        gail_agent.enjoy()

    main(args)



