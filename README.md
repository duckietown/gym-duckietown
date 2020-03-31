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
python test.py --episodes 9 --pretrain 1 --training-name "pretrained9"
python test.py --episodes 1 --pretrain 1 --training-name "pretrained1"
python test.py --episodes 3 --pretrain 1 --training-name "pretrained3"
python test.py --episodes 7 --pretrain 1 --training-name "pretrained7"
```
# Gail Lite
```bash
python test.py --episodes 1 --pretrain 0 --training-name "gaillite1" --checkpoint "pretrained1_epoch_200"
python test.py --episodes 3 --pretrain 0 --training-name "gaillite3" --checkpoint "pretrained3_epoch_200"
python test.py --episodes 7 --pretrain 0 --training-name "gaillite7" --checkpoint "pretrained7_epoch_200"
python test.py --episodes 9 --pretrain 0 --training-name "gaillite9" --checkpoint "pretrained9_epoch_200" --pretrain-D 200
```