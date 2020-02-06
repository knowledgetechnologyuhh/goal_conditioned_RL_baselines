from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.util import (
        import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.common.mpi_adam import MpiAdam
from baselines.template.policy import Policy
from baselines.hac.layer import Layer

from baselines.hac.agent import Agent
import baselines.hac.env_designs
from baselines.hac.options import parse_options
import os,sys,inspect
import importlib


class HACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, polyak, batch_size,
            Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
            rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
            sample_transitions, gamma, reuse=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
        self.FLAGS = parse_options()
        self.FLAGS.mix_train_test = True
        self.FLAGS.retrain = True
        self.FLAGS.Q_values = True

        agent, env = self.init_levy(self.FLAGS)
        wtm_agent, wtm_env, self.FLAGS = self.wtm_env_levy_style(kwargs['make_env'], self.FLAGS)
        self.check_envs(env, wtm_env)

        self.agent = wtm_agent
        self.env = wtm_env
        #  self.agent = agent
        #  self.env = env

    def check_envs(self, env, wtm_env):
        #  assert env.model == wtm_env.model
        assert env.name == wtm_env.name
        assert type(env.sim) == type(wtm_env.sim)
        assert env.state_dim == wtm_env.state_dim
        assert env.action_dim == wtm_env.action_dim
        assert (env.action_bounds == wtm_env.action_bounds).all()
        assert (env.action_offset == wtm_env.action_offset).all()
        assert env.end_goal_dim == wtm_env.end_goal_dim
        assert env.subgoal_dim == wtm_env.subgoal_dim
        assert (env.subgoal_bounds == wtm_env.subgoal_bounds).all()
        assert (env.subgoal_bounds_symmetric == wtm_env.subgoal_bounds_symmetric).all()
        assert (env.subgoal_bounds_offset == wtm_env.subgoal_bounds_offset).all()
        assert env.max_actions == wtm_env.max_actions
        assert(env.initial_state_space == wtm_env.initial_state_space).all()
        assert(env.end_goal_thresholds == wtm_env.end_goal_thresholds).all()
        assert(env.subgoal_thresholds == wtm_env.sub_goal_thresholds).all()
        assert env.goal_space_train == wtm_env.goal_space_train
        assert env.goal_space_test == wtm_env.goal_space_test
        assert(env.subgoal_bounds == wtm_env.subgoal_bounds).all()

        print('PASSED ASSERTS')

    def wtm_env_levy_style(self,make_env, FLAGS):
        env = make_env().env
        print(type(env), dir(env))

        # design_agent_and_env
        FLAGS.layers = 2
        if FLAGS.time_scale == 0:
            # Enter max sequence length in which each policy will specialize
            FLAGS.time_scale = 30

        max_actions = 700
        max_actions = FLAGS.time_scale**(FLAGS.layers)
        timesteps_per_action = 15
        env.max_actions = max_actions
        env.visualize = False

        env.state_dim = self.input_dims['o']

        env.action_dim = len(env.sim.model.actuator_ctrlrange)
        env.action_bounds = env.sim.model.actuator_ctrlrange[:,1]
        env.action_offset = np.zeros((len(env.action_bounds)))

        # different naming
        env.project_state_to_subgoal = env.project_state_to_sub_goal
        env.subgoal_thresholds = env.sub_goal_thresholds

        env.end_goal_dim = len(env.goal_space_test)
        env.subgoal_dim = len(env.subgoal_bounds)
        print('dims: action = {}, subgoal = {}, end_goal = {}'.format(env.action_dim, env.subgoal_dim, env.end_goal_dim))

        env.subgoal_bounds_symmetric = np.zeros((len(env.subgoal_bounds)))
        env.subgoal_bounds_offset = np.zeros((len(env.subgoal_bounds)))
        for i in range(len(env.subgoal_bounds)):
            env.subgoal_bounds_symmetric[i] = (env.subgoal_bounds[i][1] - env.subgoal_bounds[i][0])/2
            env.subgoal_bounds_offset[i] = env.subgoal_bounds[i][1] - env.subgoal_bounds_symmetric[i]

        print('subgoal_bounds: symmetric {}, offset {}'.format(env.subgoal_bounds_symmetric, env.subgoal_bounds_offset))

        env.get_next_goal = env._sample_goal
        env.reset_sim = env._reset_sim

        def exe_action(action):
            env.sim.data.ctrl[:] = action
            env.sim.step()
            #  env._set_action(action)

            if env.visualize:
                env.render()

            return env._get_state()

        env.execute_action = exe_action
        env.velo_threshold = 0.8

        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["subgoal_penalty"] = -FLAGS.time_scale
        agent_params["atomic_noise"] = [0.1 for i in range(8)]
        agent_params["subgoal_noise"] = [0.1 for i in range(len(env.sub_goal_thresholds))]
        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100
        FLAGS.id = 1
        agent = Agent(FLAGS,env,agent_params)

        return agent, env, FLAGS


    def init_levy(self, FLAGS):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir)
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        env_import_name = "baselines.hac.env_designs.ANT_FOUR_ROOMS_2_design_agent_and_env"
        design_agent_and_env_module = importlib.import_module(env_import_name)
        # simple tag for agent's tf scope
        FLAGS.id = 0
        # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
        agent, env = design_agent_and_env_module.design_agent_and_env(FLAGS)
        return agent, env
