from baselines.util import (store_args)
from baselines.template.policy import Policy

from baselines.hac.agent import Agent
import baselines.hac.env_designs
from baselines.hac.options import parse_options
from baselines.hac.utils import EnvWrapper, check_envs, check_validity
import os,sys,inspect
import importlib

class HACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, polyak, batch_size,
            Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
            rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
            sample_transitions, gamma, reuse=False, levy_env=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
