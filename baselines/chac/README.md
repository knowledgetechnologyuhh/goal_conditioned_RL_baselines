# Curious Hierarchical Actor-Critic (CHAC)

CHAC is a combination of Levy's Hierarchical Actor-Critic (HAC, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948))
and curiosity-driven exploration ([Curiosity-driven Exploration by Self-supervised Predicton](https://pathak22.github.io/noreward-rl/).
This folder contains a HAC implementation using [PyTorch](https://pytorch.org/).
For training with curiosity, a forward model is added to each level of the hierarchy to partially add an intrinsic reward to the external reward
of the environment.

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

We can visualize the trained agent in evaluation mode by using  `--n_train_rollouts 0` and `--render 1`.

```bash
python experiment/train.py --algorithm baselines.chac --render 0 --env AntReacherEnv-v1 \
  --n_epochs 10 --n_train_rollouts 100 --restore_policy <path>/policy_latest.pkl
```
