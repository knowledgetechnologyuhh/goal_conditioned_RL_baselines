from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, prob_dist2discrete)
from baselines.herhrl.normalizer import Normalizer
from baselines.herhrl.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.template.policy import Policy
from baselines.herhrl.obs2preds import Obs2PredsModel, Obs2PredsBuffer
# from baselines.her_pddl.pddl.pddl_util import obs_to_preds_single



def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class HRL_Policy(Policy):
    @store_args
    def __init__(self, input_dims, T,
                 rollout_batch_size, child_policy=None, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer, reward function is called here
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        self.child_policy = child_policy
        self.envs = []

    def set_envs(self, envs):
        self.envs = envs
        if self.child_policy is not None:
            self.child_policy.set_envs(envs)