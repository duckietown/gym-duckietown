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

VAE vs Resnet

```bash
python main.py --enjoy 0 --eval 0 --train 0 --train-vae 1 --env_name 'duckietown'

python main.py --training-name imitate-vae-1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 1e-5 --episodes 1 --pretrain-name imitate-vae-1 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1 --D-iter 5 --use-vae 1 --imitation 1

python main.py --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.005 --env_name duckietown --update-d WGAN --lrG 1e-6 --episodes 1 --pretrain-name imitate-resnet50-1 --sampling-eps 3 --pretrain-D 200 --D-iter 5 --use-vae 0 --use-checkpoint 0 --train 1 --training-name imitate-resnet50-1 --imitation 1
```



Policy Gradient Vs PPO Vs Vanilla Returns

```bash
python main.py --update-with "POLICY GRADIENT" --eval 1 --train 1 --env_name duckietown --update-d WGAN --lrG 1e-4 --episodes 1 --pretrain-name imitate-resnet50-1 --sampling-eps 50 --ppo-steps 100 --use-vae 0 --use-checkpoint 1 --train 1 --training-name gail-pg-wgan-resnet50-1 --pretrain-D 5

python main.py --update-with "PPO" --eval 1 --train 1 --env_name duckietown --update-d WGAN --lrG 1e-4 --episodes 1 --pretrain-name imitate-resnet50-1 --sampling-eps 50 --ppo-steps 100 --use-vae 0 --use-checkpoint 1 --train 1 --training-name gail-ppo-wgan-resnet50-1 --ppo-epochs 1 --pretrain-D 500

python main.py --update-with "0" --eval 1 --train 1 --env_name duckietown --update-d WGAN --lrG 1e-4 --episodes 1 --pretrain-name imitate-resnet50-1 --sampling-eps 50 --ppo-steps 100 --use-vae 0 --use-checkpoint 1 --train 1 --training-name gail-ppo-wgan-resnet50-1 --ppo-epochs 1 --pretrain-D 500
```