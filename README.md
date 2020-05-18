# Getting started
We have some environments using the MuJoCo simulator and some with CoppeliaSim & PyRep.
You can choose to use both or only one of them.

If you want to use both:

1. Download MuJoCo (mujoco.org) and obtain a license (as a student you can obtain a free one-year student license). 
Copy the mjpro200_linux folder from the downloaded archive as well as mjkey.txt that you will obtain from the 
registration to folders of your choice.
2. Download CoppeliaSim [here](https://www.coppeliarobotics.com/ubuntuVersions) and start it to check whether it works.
3. Then `git clone https://github.com/stepjam/PyRep.git`. 
If you don't want to have to adjust the paths in `set_paths.sh`, put 
CoppeliaSim and PyRep in /data/*username*/.
You are free to put them somewhere else, but you'll have to adjust the paths then.
4. Set the environment variables for MuJoCo and CoppeliaSim in `set_paths.sh` according to the locations 
where you saved the mjpro200_linux folder, mjkey.txt and CoppeliaSim. 
If you are using an IDE, set the variables there as well. 
(Note that PyCharm does not dynamically evaluate environment variables at all, so things like `$(whoami)` 
or even `~/` will not work.)
5. Set up a virtual environment using `virtualenv -p python3 venv`
6. Activate the virtual environment using `source venv/bin/activate`
7. Install python libraries using `pip3 install -r requirements_gpu.txt` if you have a GPU or `pip3 install -r requirements.txt` if you don't have a GPU.
8. Pip install PyRep by running: 
`pip install git+https://github.com/stepjam/PyRep.git` You can find some troubleshooting on the PyRep git-page.
9. Now you can test the MuJoCo installation by running
`experiment/train.py`
and the CoppeliaSim & PyRep installation by running
`experiment/train.py --env CopReacherEnv-ik1-v0 --algorithm baselines.her`

If you want to use MuJoCo only:
 
1. Download MuJoCo (mujoco.org) and obtain a license (as student you can obtain a free one-year student license). Copy the mjpro200_linux folder from the downloaded archive as well as mjkey.txt that you will obtain from the registration to folders of your choice
2. Set the environment variables in `set_paths.sh` according to the locations where you saved the mjpro200_linux folder and the mjkey.txt. 
In `set_paths.sh`, comment out the lines below `#For CoppeliaSim`.
If you are using an IDE, set the variables there as well. 
(Note that PyCharm does not dynamically evaluate environment variables at all, so things like `$(whoami)` 
or even `~/` will not work.)
3. Set up a virtual environment using `virtualenv -p python3 venv`
4. Activate the virtual environment using `source venv/bin/activate`
5. Install python libraries using `pip3 install -r requirements_gpu.txt` if you have a GPU or `pip3 install -r requirements.txt` if you don't have a GPU.
6. Run script with `experiment/train.py`

If you want to use CoppeliaSim only:

1. Download CoppeliaSim [here](https://www.coppeliarobotics.com/ubuntuVersions) and start it to check whether it works.
2. Then `git clone https://github.com/stepjam/PyRep.git`. 
If you don't want to have to adjust the paths in `set_paths.sh`, put 
CoppeliaSim and PyRep in /data/*username*/.
You are free to put them somewhere else, but you'll have to adjust the paths then.
3. In `set_paths.sh`, comment out the lines below `#For MuJoCo` and, if necessary, edit the lines below `#For CoppeliaSim`.
If you are using an IDE, set the variables there as well. 
(Note that PyCharm does not dynamically evaluate environment variables at all, so things like `$(whoami)` 
or even `~/` will not work.)
4. Set up a virtual environment using `virtualenv -p python3 venv`
5. Activate the virtual environment using `source venv/bin/activate`
6. Install python libraries using `pip3 install -r requirements_gpu.txt` 
if you have a GPU or `pip3 install -r requirements.txt` if you don't have a GPU, 
but comment out `mujoco-py==2.0.2.5` in the respective requirements.txt before.
7. Pip install PyRep by running: 
`pip install git+https://github.com/stepjam/PyRep.git` You can find some troubleshooting on the PyRep git-page.
8. Now you can test the MuJoCo installation by running
`experiment/train.py`
and the CoppeliaSim & PyRep installation by running
`experiment/train.py --env CopReacherEnv-ik1-v0 --algorithm baselines.her`

<br/><br/>
<br/><br/>

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


# Adding new features to this repository
If you work on this repository and add a new feature, please proceed as follows:
1. Start a new branch with the devel branch as base and add your feature

    * If you develop a new algorithm, you should have added a respective subdir with the algorithm's name 
    (referred to here as <alg_name>) in the baselines folder.
    In *baselines/templates/* you can find templates that you can use for your algorithm.
    * If your algorithm is finished, proceed as follows: 
        * In the `generate_testing_commands.py` script, add the <alg_name> to TestingConfig.algorithms, and 
        * add a function get_<alg_name>_cmds to this script. This function should generate a list of commandline-strings 
        that call all important configurations for your algorithm.
    
    * If you add a feature to an existing algorithm, only change the get_<alg_name>_cmds function appropriately.
    
    * If you add a new environment, add the environment name to the TestingConfig.environments list.
      You find tips for adding a new environment further below.

2. After you have finished working on your feature:
    * merge the devel branch to your branch and 
    * run the testing script `run_testing.sh`. The script will create a folder testing_logs where all test results are stored.
    * Go through all logs and see if there are errors. If there are errors, fix them.
3. If all errors are fixed, check if there are new updates on devel. 
    * If there are, goto 2. 
    * Else: merge your branch to devel. Done.   


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

**If you want to use CoppeliaSim**, create your environment in *wtm_envs/coppelia/*.
Look at the *cop_reach_env* to see which functions are needed so that the environment
can be used by the algorithms. At the top of *cop_reach_env.py* a scene file is referenced.
You will also need a CoppeliaSim scene file (ends with .ttt) for your environment. You can
edit the .ttt file in the CoppeliaSim IDE. Use PyRep to control objects in the CoppeliaSim scene.

**Independent of using MuJoCo or CoppeliaSim**, you'll have to register your environment in 
*wtm_envs/register_envs.py*.
You use the environment placed in *wtm_envs/mujoco/<environment_name>/<task_name>.py for this.

