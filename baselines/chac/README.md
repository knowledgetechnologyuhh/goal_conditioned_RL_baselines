# Curious Hierarchical Actor-Critic (CHAC)

CHAC is a combination of Levy's (HAC, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948))
and curiosity-driven exploration. This folder contains a HAC implementation using PyTorch.
For training with curiosity, a forward model is added to each level of the hierarchy to partially
update the reward for training the critic.

Further details can be found in our paper [Curious Hierarchical Actor-Critic Reinforcement Learning](https://arxiv.org/abs/2005.03420).


## Train with HAC
```bash
python experiment/train.py --algorithm baselines.chac --render 0 --env AntReacherEnv-v0 \
  --n_epochs 10 --n_train_rollouts 100
```

## Train with CHAC

```bash
python experiment/train.py --algorithm baselines.chac --render 0 --env AntReacherEnv-v0 \
  --n_epochs 10 --n_train_rollouts 100 --fw 1 --fw_hidden_size 256,256,256 --eta 0.5
```

## Restore a saved policy

This is also useful to see how the agent performs and currently shows the training
mode and not exploration only.

```bash
python experiment/train.py --algorithm baselines.chac --render 1 --env AntReacherEnv-v1 --graph 0 --restore_policy <path>/policy_latest.pkl
```
