# Getting started

1. Download MuJoCo (mujoco.org) and obtain a license (as student you can obtain a free one-year student license). Copy the mjpro200_linux folder from the downloaded archive as well as mjkey.txt that you will obtain from the registration to folders of your choice
2. Set the environment variables in `set_paths.sh` according to the locations where you saved the mjpro200_linux folder and the mjkey.txt. If you are using an IDE, set the variables there as well. 
3. Set up virtual environment using `virtualenv -p python3 venv`
4. Activate virtualenvironment using `source venv/bin/activate`
5. Install python libraries using `pip3 install -r requirements_gpu.txt` if you have a GPU or `pip3 install -r requirements.txt` if you don't have a GPU.
6. Run script with `experiment/train.py`

Logs will be stored in a directory according to the `--base_logdir` command line parameter (by default `data`). It will create a subdirecory according to the git commit id and then a subdirectory according to the number of trials the experiment with the same parameters has been performed so far.

# Currently supported algorithms
The algorithm can be selected using the command line option `--algorithm` (see below).

Algorithm-specific implementation details are stored in `baselines/<alg name>`.
We currently support `baselines.her` (Hindsight Experience Replay) as comparison and baseline to our results.
We also support `baselines.herhrl` (Hindsight Experience Replay with Hierarchical Reinforcement Learning).
We have experimental support for `baselines.model_based` which learns a model of the environment but does not yet generate useful actions.

# Command line options
Command line options are realized using the *click* library. They should be more or less self-explanatory.
General command line options can be found in `experiment/click_options.py`

Algorithm-specific command line options can be found in `baselines/<alg name>/interface/click_options.py`


# herhrl algorithm
This is adapted from the implementation by Levy et al. 2018. We extended it with the possibility to also use a high-level PDDL planner to generate subgoals. However, PDDL requires the implementation of an environment-specific file to generate planning domains. An example is `wtm_envs/mujoco/tower_env_pddl.py`, which extends `wtm_envs/mujoco/tower_env.py` with PDDL capabilities. We also support a mixed HRL/PDDL policy which switches over from PDDL to HRL when PDDL does not continue to improve the success rate. The following parameters are most important:

* `--policies_layers` determines the layers above the lower level layer. The low-level layer is always `DDPG_HER_HRL_POLICY`. That is, if you define `--policies_layers [DDPG_HER_HRL_POLICY]` then this means that you have a two-layer HRL agent with the high level layer being DDPG_HER_HRL_POLICY and the low-level layer also DDPG_HER_HRL_POLICY. If you set `--policies_layers [PDDL_POLICY,DDPG_HER_HRL_POLICY]` then you have a three-layer HRL agent with the high level layer being PDDL, the mid-level layer being  DDPG_HER_HRL_POLICY and the low-level layer also being DDPG_HER_HRL_POLICY. However, we have not yet tested more than two layers. It is, therefore, highly recommended to use no more than two layers in total at this point of the implementation, i.e., either set `--policies_layers [DDPG_HER_HRL_POLICY]` or `--policies_layers [PDDL_POLICY]`

* `--n subgoals_layers` determines the max. amount of steps allowed for all layers except for the lowest layer. For example, `--n subgoals_layers [10]` means that you have a two-layer agent where the high-level layer has at most 10 steps.

* `--penalty_magnitude` determines the penalty score when the subgoal is missed, as described in Levy et al. 2018

* `--test_subgoal_perc` determines the probability that a penalty is applied to a specific high-level policy step (also see Levy et al. 2018).


# model_based algorithm
1. Initialize replay buffer (The replay buffer is implemented in the class [ModelReplayBuffer](baselines/model_based/mb_policy.py) and initialized as part of the class [MBPolicy](baselines/model_based/mb_policy.py).)
2. **Exploration Phase**
    1. For each epoch (in function [train](experiment/train.py))
        1. For each episode (in function  [generate_rollouts_update](baselines/model_based/rollout.py) in class `RolloutWorker`)
            1. Generate Rollouts: For each step (in function `generate_rollouts` in [Rollout](baselines/template/rollout.py)
                1. policy.get_actions (in function [MBPolicy](baselines/model_based/mb_policy.py))
                1. Return rollouts
            1. `policy.store_episode` (This is implemented in class [ModelReplayBuffer](baselines/model_based/model_replay_buffer.py) May throw away other rollouts if they are not interesting anymore. The computation of the "level of interestingness", represented by variable `ModelReplayBuffer.memory_value`, is determined by command line argument `--memval_method`)
            1. For each sampled train_batch: (batch sampling is realized by command line argument `buff_sampling', default is "random")
                1. `losses = policy.train()`
                1. Recompute `ModelReplayBuffer.memory_value` for those samples that were included in the batch.
            1. Execute `test_prediction_error` to count the number of successive forward modeling steps that are possible without moving away too much from ground truth. (Maybe good for evaluating when to move from exploration to exploitation)
            
3. **Exploitation Phase**
    As in HER, but instead of generating actions in `policy.get_actions`, generate multiple candidate-actions (e.g. by adding noise to observations) and use the model to select the most promising one.



