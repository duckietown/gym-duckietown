import argparse

from gail.train import _train
from learning.imitation.basic.enjoy_imitation import _enjoy
from learning.imitation.basic.train_imitation import _train as t2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=1, type=int, help="Train new model")
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=9, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=50, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=10000, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")
    parser.add_argument("--data-directory", default="D:/Michael/Learning/duckietown_data/", type=str, help="Where to save generated expert data")
    parser.add_argument("--lrG", default=0.004, type=float, help="Generator learning rate")
    parser.add_argument("--lrD", default=0.004, type=float, help="Discriminator learning rate")
    parser.add_argument("--get-samples", default=0, type=int, help="Generate expert data")
    parser.add_argument("--use-checkpoint", default=0, type=int, help="Use checkpoint for training")
    parser.add_argument("--checkpoint", default="best", type=str, help="file name for checkpoint for training")
    parser.add_argument("--enjoy_tag", default="_best", type=str, help="file tag for checkpoint for enjoying")



    args = parser.parse_args()

    if args.train:
        print("let's train!")
        _train(args)
        # t2(args)
    else:
        _enjoy(args)
