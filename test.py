import argparse

from gail.train import _train
from learning.imitation.basic.enjoy_imitation import _enjoy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int, help="Train new model")
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=80, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=9, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=24000, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")
    parser.add_argument("--data-directory", default="D:/Michael/Learning/duckietown_data/", type=str, help="Where to save generated expert data")
    parser.add_argument("--lrG", default=0.004, type=float, help="Generator learning rate")
    parser.add_argument("--lrD", default=0.004, type=float, help="Discriminator learning rate")
    parser.add_argument("--get-samples", default=1, type=int, help="Generate expert data")
    parser.add_argument("--use-checkpoint", default=0, type=int, help="Use checkpoint for training")
    parser.add_argument("--checkpoint", default="best", type=str, help="file name for checkpoint for training")


    args = parser.parse_args()

    if args.train:
        print("let's train!")
        _train(args)
    else:
        _enjoy()
