# Getting started
We have some environments using the MuJoCo simulator and some with CoppeliaSim & PyRep.
You can choose to use both or only one of them.

1. If you would like to use the MuJoCo-based environments:
    1. Download MuJoCo (mujoco.org) and obtain a license (as a student you can obtain a free one-year student license). 
Copy the mjpro200_linux folder from the downloaded archive as well as mjkey.txt that you will obtain from the 
registration to folders of your choice.
    2. Set the environment variables for MuJoCo in `set_paths.sh` according to the locations 
where you saved the mjpro200_linux folder and mjkey.txt.
1. If you would like to use the Coppelia-based environments: 
    1. Download CoppeliaSim [here](https://www.coppeliarobotics.com/ubuntuVersions) and start it to check whether it works.
    1. Set the environment variable COPPELIASIM_ROOT in `set_paths.sh` according to the location of your CoppeliaSim installation.
1. Run `source set_paths.sh`
3. Set up a virtual environment using `virtualenv -p python3 venv`
4. Activate the virtual environment using `source venv/bin/activate`
5. Install python libraries using `pip3 install -r requirements_gpu.txt` if you have a GPU or `pip3 install -r requirements.txt` if you don't have a GPU.
Comment out `mujoco-py` in the *requirements.txt* if you are not using MuJoCo.
6. If you would like to use the Coppelia-based environments, Pip install PyRep by running: 
`pip install git+https://github.com/stepjam/PyRep.git`. You can find some troubleshooting on the PyRep git-page.
7. You can test the MuJoCo installation by running
`experiment/train.py`
and the CoppeliaSim & PyRep installation by running
`experiment/train.py --env CopReacherEnv-ik0-v0 --algorithm baselines.her`

Logs will be stored in a directory according to the `--base_logdir` command line parameter (by default `data`). It will create a subdirecory according to the git commit id and then a subdirectory according to the number of trials the experiment with the same parameters has been performed so far.

# Currently supported algorithms
The algorithm can be selected using the command line option `--algorithm` (see below).

Algorithm-specific implementation details are stored in `baselines/<alg name>`. We currently support the following algorithms: 
 
 * `baselines.her` (Hindsight Experience Replay) as comparison and baseline to our results. The HER code is copied from the [OpenAI baselines repository](https://github.com/openai/baselines)
 * `baselines.chac` (Curious Hierarchical Actor Critic). This extends the Hierarchical Actor Critic approach by Levy et al. (2019) with a curiosity-based mechanism (see https://arxiv.org/abs/2005.03420). In contrast to Levy's HAC which uses Tensorflow, our implementation is based on PyTorch.
  
 * `baselines.herhrl` (Hindsight Experience Replay with Hierarchical Reinforcement Learning), but in its current implementation this is known to provide poor performance results. 

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
- `--n_levels 1-3` to specify the numbers of hierarchies used
- `--time_scales 25,10` to determine the number of actions each level is allowed to do (from lowest to highest as comma-separated string)
- `--fw 0/1` to enable training with curiosity
- `--fw_hidden_layer 256,256,256` a comma-separated string to specify depth and size
- `--eta 0.5` specifies how much of the intrinsic reward to use and external reward to keep

For further details consider looking at the CHAC Readme [baselines/chac/README.md](./baselines/chac/README.md).

# Adding new features to this repository
If you work on this repository and add a new feature, please proceed as follows:
1. Start a new branch with the devel branch as base and add your feature

    * If you develop a new algorithm, you should have added a respective subdir with the algorithm's name 
    (referred to here as <alg_name>) in the baselines folder.
    * In *baselines/example_algorithm/* you find an example algorithm with a dummy policy that
    generates random actions and does not learn. You can use this as a starting point.
    * In *baselines/templates/* you can find templates that you can use for your algorithm.
    * If your algorithm is finished, proceed as follows: 
        * In the `generate_testing_commands.py` script, add the <alg_name> to TestingConfig.algorithms, and 
        * add a function get_<alg_name>_cmds to this script. This function should generate a list of commandline-strings 
        that call all important configurations for your algorithm.
    
    * If you add a feature to an existing algorithm, only change the get_<alg_name>_cmds function appropriately.

    * If you add a new environment, add the environment name to the TestingConfig.environments list.
      You find tips for adding a new environment further below.

2. After you have finished working on your feature:
    * Run the testing script `run_testing.sh`. The script will create a folder testing_logs where all test results are stored.
    * Go through all logs and see if there are errors. If there are errors, fix them.


## Tips for adding a new environment
Currently there are two types of environments in this project. 
Ones that use MuJoCo and ones that use CoppeliaSim.

**If you want to use MuJoCo**, let your environment class inherit from *wtm_envs/mujoco/wtm_env.py*.
Also take a look at the other environments in the folder, e.g. *blocks_env*, to see which functions are needed so that the algorithms
can operate on the environment. 
Put your <environment_name>.py file into *wtm_envs/mujoco/*.
In *wtm_envs/mujoco/assets/*, create a subfolder with your environment name 
and save your *environment.xml* in it.

In *wtm_envs/mujoco/*, create a subfolder with your environment name and 
put an empty file *\_\_init\_\_.py* in it.
In the same folder, create a python script with a name that resembles the task that the
agent should solve in your environment. In it, create a class that inherits from your 
environment class (the one that you have in *wtm_envs/mujoco/<environment_name>.py*) and
set some task-specific parameters there.

Then register your environment in *wtm_envs/register_envs.py*.
You use the environment placed in *wtm_envs/mujoco/<environment_name>/<task_name>.py* for this.

**If you want to use CoppeliaSim**, create your environment in *wtm_envs/coppelia/*.
Look at the *cop_reach_env* to see which functions are needed so that the environment
can be used by the algorithms. At the top of *cop_reach_env.py* a scene file is referenced.
You will also need a CoppeliaSim scene file (ends with .ttt) for your environment. You can
edit the .ttt file in the CoppeliaSim IDE. Use PyRep to control objects in the CoppeliaSim scene.

Then add a handler class for your environment, like the `ReacherEnvHandler` in 
*wtm_envs/coppelia/cop_reach_env*. 
Register your environment-handler in *wtm_envs/register_envs.py*.
