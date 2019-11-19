
ls -l ./reinforcement/pytorch/models/ddpg_actor.pth
ls -l ./reinforcement/pytorch/models/ddpg_critic.pth

python3 -m  imitation.pytorch.enjoy_imitation  | tee enjoy_imitation.log ; say ok


