import argparse
import re

from gail.train import _train
from gail.eval import _eval
from learning.imitation.basic.enjoy_imitation import _enjoy
from learning.imitation.basic.train_imitation import _train as _train2
if __name__ == '__main__':
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
    parser.add_argument("--D-train-eps", default=5, type=int, help="number of times to train D vs G, negative means train D every G updates")
    parser.add_argument("--pretrain", default=0, type=int, help="flag to run imitation learning instead")
    parser.add_argument("--eval", default=1, type=int, help="flag to run eval script")
    parser.add_argument("--eps", default=0.000001, type=float, help="epsilon for imitation learning")
    parser.add_argument("--eval-steps", default=5, type=int, help="number of steps to evaluate policy on")
    parser.add_argument("--eval-episodes", default=50, type=int)
    parser.add_argument("--enjoy", default=0, type=int)
    parser.add_argument("--pretrain-D", default=50, type=int)


    parser.add_argument("--gamma", default=0.995, type=float)
    parser.add_argument("--lam", default=0.97, type=float)
    parser.add_argument("--clip-param", default=0.2, type=float)
    parser.add_argument("--entropy-beta", default=0.001, type=float)
    parser.add_argument("--ppo-epsilon", default=0.02, type=float)
    parser.add_argument("--ppo-epochs", default=5, type=int)
    parser.add_argument("--ppo-steps", default=256, type=int)
    parser.add_argument("--critic-discount", default=0.5, type=float)
    parser.add_argument("--do-ppo", default=0, type=int)


    args = parser.parse_args()


    experiments = ["pretrained9",
                    "pretrained7",
                    "pretrained3",
                    "pretrained1",
                    "gaillite1",
                    "gaillite3",
                    "gaillite7",
                    "gaillite9",
                    ]

    # scores = {}
    # for name in experiments:
    #     args.training_name = name
    #     args.episodes = int(re.sub(r'[a-z]+', '', name, re.I))
    #     if "gail" in name:
    #         args.checkpoint = "pretrained1"
    #         _train(args)
    #         scores[name] = _eval(args)

    #         args.rollout = 0
    #         _train(args)
    #         scores[name+"norollout"] = _eval(args)           
    #     else:
    #         _train2(args)
    #         scores[name] = _eval(args)

    # torch.save(scores, "scores.pt")


    # scores = torch.load("scores.pt")


    if args.train:
        print("let's train!")
        if args.pretrain:
            from learning.imitation.basic.train_imitation import _train 
        _train(args)
    if args.eval:

        _eval(args)
    if args.enjoy:
        _enjoy(args)
