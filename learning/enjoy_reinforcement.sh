
ls -l ./reinforcement/pytorch/models/ddpg_actor.pth
ls -l ./reinforcement/pytorch/models/ddpg_critic.pth

python3 -m reinforcement.pytorch.enjoy_reinforcement  | tee enjoy.log ; say ok


