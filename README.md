# Getting started

1. Install MuJoCo (mujoco.org) and copy mjpro150 folder as well as mjkey.txt to ~/.mujoco
2. Set environment variables according to your Graphics driver version as in `set_paths.sh`
3. Set up virtual environment using `virtualenv -p python3 venv`
4. Activate virtualenvironment using `source venv/bin/activate`
5. Install python libraries using `pip3 install -r requirements_gpu.txt`
6. Run script with `experiment/train.py`

Logs will be stored in a directory according to the `--logs` command line parameter (by default `logs`). It will create a subdirecory according to the git commit id and then a subdirectory according to the number of trials the experiment with the same parameters has been performed so far. The name of this second subdirectory is determined in the `main` function of `train.py`, according to line `override_params = config.OVERRIDE_PARAMS_LIST`.



# Currently supported algorithms
Algorithm-specific implementation details are stored in `baselines/<alg name>`.
We currently support `baselines.her` (Hindsight Experience Replay) as comparison and baseline to our results. This code is about `baselines.model_based`. Hence, the folder `baselines/model_based` is where most of the coding takes place. The algorithm can be selected using the command line option `--algorithm` (see below).

# Command line options
Command line options are realized using the *click* library.
General command line options can be found in `experiment/click_options.py`

Algoritm specific command line options can be found in `baselines/<alg name>/interface/click_options.py`


# Main algorithm
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

# TODO
1. Currently, the action selection in the exploration phase is random, and the surprise prediction is used to select the action out of a candidate set of actions that is supposed to maximize the model error (kind of beam search). Instead of this approach, use another RL algorithm, e.g. PPO, to maximize the model loss during exploration. We can later combine that algorithm also with the model, but this should happen in a second step.
2. Implement Exploitation Phase

