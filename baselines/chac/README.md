# Curious Hierarchical Actor-Critic (CHAC)

CHAC is a combination of Levy's Hierarchical Actor-Critic (HAC, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948))
and curiosity-driven exploration ([Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/)).
This folder contains a HAC implementation using [PyTorch](https://pytorch.org/).
For training with curiosity, a forward model is added to each level of the hierarchy to partially add an intrinsic reward to the external reward
of the environment.

Further details can be found in the paper [Curious Hierarchical Actor-Critic Reinforcement Learning](https://arxiv.org/abs/2005.03420).

## Getting Started

### Train with HAC (without curiosity)
```bash
python experiment/train.py --algorithm baselines.chac --render 0 --env AntReacherEnv-v0 \
  --n_epochs 10 --n_train_rollouts 100 --fw 0
```

### Train with CHAC (with curiosity)

```bash
python experiment/train.py --algorithm baselines.chac --render 0 --env AntReacherEnv-v0 \
  --n_epochs 10 --n_train_rollouts 100 --fw 1 --fw_hidden_size 256,256,256 --eta 0.5
```

### Restore a saved policy

Visualize the trained agent in evaluation mode with `--n_train_rollouts 0` and `--render 1`.

```bash
python experiment/train.py --algorithm baselines.chac --render 0 --env AntReacherEnv-v1 \
  --n_epochs 10 --n_train_rollouts 100 --restore_policy <path>/policy_latest.pkl
```

## Click Options in Detail

### `--fw 1`

This option includes the forward model into the training process and enables
curiosity for training the actor-critic. When set to `0`, CHAC is effectively the same as HAC and should provide comparable results.

### `--eta 0.5`

Determines the balance between external reward r_e and curiosity r_c.
> reward = r_e * eta + (1 - eta) * r_c.

### `--fw_hidden_size 256,256,256`

Provide a string of comma-separated values like `256,256,256` to create three
hidden layers, each with 256 units. This parameter is crucial in defining the
capability of the forward model to learn the transition function of the MDP.

### `--n_levels 2`

This option defines the number of hierarchies. For easy environments,
`1` level can be used to receive fast results.

### `--time_scales 25,10`

This is a comma-separated list of integers to define how many actions each level
of the hierarchy is allowed to do until another routine is initiated. For `1` level
(`--n_levels 1`), it is required to also provide one integer. In this case, the value will not
be used as action limit but is important for the Subgoal Testing Transitions penalty (cf HAC).

### `--subgoal_test_perc 0.3`

The probability to test an upcoming subgoal via Subgoal Testing (cf HAC).
