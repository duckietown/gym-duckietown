## Installation
```bash
git clone git@github.com:mphamhung/gym-duckietown.git
cd gym-duckietown
conda env create -f environment.yaml
conda activate gail-env
```
## add conda env to jupyter
```bash
python -m ipykernel install --user --name=gail-env
```

## start tensorboard
```bash
cd gym-duckietown
tensorboard --logdir=runs
```

## start jupyter 
```bash
jupyter notebook
```

## Gym-DuckieTOWN

Read [this](https://github.com/mphamhung/gym-duckietown/blob/master/README_old.md) for duckietown readme


## Experiments run

# Imitation Learning
```bash
python test.py --episodes 1 --steps 100 --batch-size 100 --pretrain 1 --training-name "pretrained1"
python test.py --episodes 3 --steps 100 --batch-size 100 --pretrain 1 --training-name "pretrained3"
python test.py --episodes 7 --steps 100 --batch-size 100 --pretrain 1 --training-name "pretrained7"
python test.py --episodes 9 --pretrain 1 --training-name "pretrained9"
```
# Gail Lite
```bash
python test.py --episodes 1 --steps 50 --pretrain 0 --training-name "gaillite1" --checkpoint "pretrained1"
python test.py --episodes 3 --steps 50 --pretrain 0 --training-name --checkpoint "pretrained3"
python test.py --episodes 7 --steps 50 --pretrain 0 --training-name --checkpoint "pretrained7"
python test.py --episodes 9 --steps 50 --pretrain 0 --training-name --checkpoint "pretrained9"
```