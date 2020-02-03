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


        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir)

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
        self.FLAGS = parse_options()
        env_import_name = "baselines.hac.env_designs.ANT_FOUR_ROOMS_2_design_agent_and_env"
        design_agent_and_env_module = importlib.import_module(env_import_name)

        # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
        self.agent, self.env = design_agent_and_env_module.design_agent_and_env(self.FLAGS)
