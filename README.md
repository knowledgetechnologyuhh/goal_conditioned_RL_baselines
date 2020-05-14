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
This is adapted from the implementation by Levy et al. 2018. We extended it with the possibility to also use a high-level PDDL planner to generate subgoals. However, PDDL requires the implementation of an environment-specific file to generate planning domains. An example is `wtm_envs/mujoco/blocks_env_pddl.py`, which extends `wtm_envs/mujoco/blocks_env.py` with PDDL capabilities. We also support a mixed HRL/PDDL policy which switches over from PDDL to HRL when PDDL does not continue to improve the success rate. The following parameters are most important:

* `--policies_layers` determines the layers above the lower level layer. The low-level layer is always `DDPG_HER_HRL_POLICY`. That is, if you define `--policies_layers [DDPG_HER_HRL_POLICY]` then this means that you have a two-layer HRL agent with the high level layer being DDPG_HER_HRL_POLICY and the low-level layer also DDPG_HER_HRL_POLICY. If you set `--policies_layers [PDDL_POLICY,DDPG_HER_HRL_POLICY]` then you have a three-layer HRL agent with the high level layer being PDDL, the mid-level layer being  DDPG_HER_HRL_POLICY and the low-level layer also being DDPG_HER_HRL_POLICY. However, we have not yet tested more than two layers. It is, therefore, highly recommended to use no more than two layers in total at this point of the implementation, i.e., either set `--policies_layers [DDPG_HER_HRL_POLICY]` or `--policies_layers [PDDL_POLICY]`

* `--n subgoals_layers` determines the max. amount of steps allowed for all layers except for the lowest layer. For example, `--n subgoals_layers [10]` means that you have a two-layer agent where the high-level layer has at most 10 steps.

* `--penalty_magnitude` determines the penalty score when the subgoal is missed, as described in Levy et al. 2018

* `--test_subgoal_perc` determines the probability that a penalty is applied to a specific high-level policy step (also see Levy et al. 2018).


# chac algorithm
This algorithm is an adapted version of Levy's Hierarchical Actor-Critic
(HAC, [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948)) and curiosity-driven exploration
inspired by [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/).
For curiosity, we add a forward model to every level of the hierarchy in order to learn the transition function
and produce a surprise at inference time. To implement our neural networks, we make use of PyTorch Version 1.5.0.

The most important options to consider are:
- `--n_levels 1-3` to specify the numbers of hierarchies used (currently 2 levels seem most useful)
- `--time_scales 25,10` to determine the number of actions each level is allowed to do (from lowest to highest as comma-separated string)
- `--fw 0/1` to enable training with curiosity
- `--fw_hidden_layer 256,256,256` as comma-separated string to specify depth and size
- `--eta 0.5` specifies how much of the internal reward to use and external reward to keep (`0.5` as good default for now, see paper)

For further details consider looking at the [baselines/chac/README.md](./baselines/chac/README.md).

# Adding new features to this repository
If you work on this repository and add a new feature, please proceed as follows:
1. Start a new branch with the devel branch as base and add your feature
    * If you develop a new algorithm, you should have added a respective subdir with the algorithm's name
    (referred to here as <alg_name>) in the baselines folder. Proceed as follows:
        * Add the <alg_name> to TestingConfig.algorithms, and
        * add a function get_<alg_name>_cmds to this script. This function should generate a list commandline-strings that call all important
    configurations for your algorithm.

    * If you add a feature to an existing algorithm, only change the get_<alg_name>_cmds function appropriately.

    * If you add a new environment, add the environment name to the TestingConfig.environments list.

2. After you have finished working on your feature:
    * merge the devel branch to your branch and
    * run the testing script `run_testing.sh`. The script will create a folder testing_logs where all test results are stored.
    * Go through all logs and see if there are errors. If there are errors, fix them.
3. If all errors are fixed, check if there are new updates on devel.
    * If there are, goto 2.
    * Else: merge your branch to devel. Done.



