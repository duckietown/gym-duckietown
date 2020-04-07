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

```bash
python main.py --training-name imitate-1 --imitation 1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 0.04 --episodes 1
python main.py --training-name gail-ppo-bce-lr1e_5-1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 1e-5 --episodes 1 --pretrain-name imitate-1 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1
python main.py --training-name gail-ppo-wgan-lr1e_5-1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d WGAN --lrG 1e-5 --episodes 1 --pretrain-name imitate-1 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1 

python main.py --training-name imitate-3 --imitation 1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 0.04 --episodes 3
python main.py --training-name gail-ppo-bce-lr1e_5-3 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 1e-5 --episodes 3 --pretrain-name imitate-3 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1

python main.py --training-name imitate-5 --imitation 1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 0.04 --episodes 5
python main.py --training-name gail-ppo-bce-lr1e_5-5 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 1e-5 --episodes 5 --pretrain-name imitate-5 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1

python main.py --training-name imitate-9 --imitation 1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 0.04 --episodes 9
python main.py --training-name gail-ppo-bce-lr1e_5-9 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 1e-5 --episodes 9 --pretrain-name imitate-9 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1


```


python main.py --training-name imitate-1 --eval 1 --train 0 --env_name duckietown
python main.py --training-name gail-ppo-bce-lr1e_5-1 --eval 1 --train 0 --env_name duckietown
python main.py --training-name imitate-3 --eval 1 --train 0 --env_name duckietown
python main.py --training-name gail-ppo-bce-lr1e_5-3 --eval 1 --train 0 --env_name duckietown


python main.py --training-name gail-ppo-wgan-lr1e_5-1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 1e-5 --episodes 1 --pretrain-name imitate-1 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1 --D-iter 5

python main.py --training-name gail-ppo-wgan-lr1e_5-3 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 1e-5 --episodes 3 --pretrain-name imitate-3 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1 --D-iter 5

python main.py --training-name gail-ppo-wgan-lr1e_5-1a --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 1e-5 --episodes 1 --pretrain-name imitate-1a --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1 --D-iter 5


python main.py --training-name imitate-9 --imitation 1 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.001 --env_name duckietown --update-d BCE --lrG 0.004 --episodes 9

python main.py --training-name gail-ppo-wgan-lr1e_5-9 --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 1e-5 --episodes 9 --pretrain-name imitate-1 --sampling-eps 3 --pretrain-D 200 --use-checkpoint 1 --D-iter 5

```


Train VAE

python main.py --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 2e-4 --episodes 1 --pretrain-name imitate-vae-1 --sampling-eps 3 --pretrain-D 200 --D-iter 5 --use-vae 1 --use-checkpoint 1 --update-with 0 --train 1 --training-name imitate-vae-1 --imitation 1

python main.py --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.00005 --env_name duckietown --update-d WGAN --lrG 1e-6 --episodes 1 --pretrain-name imitate-vae-1 --sampling-eps 3 --pretrain-D 200 --D-iter 5 --use-vae 1 --use-checkpoint 1 --train 1 --training-name gail-ppo-wgan-vae-1 

python main.py --update-with PPO --eval 1 --train 1 --epochs 1000 --d_schedule 5 --lrD 0.005 --env_name duckietown --update-d WGAN --lrG 1e-6 --episodes 1 --pretrain-name imitate-resnet50-1 --sampling-eps 3 --pretrain-D 200 --D-iter 5 --use-vae 0 --use-checkpoint 0 --train 1 --training-name imitate-resnet50-1 