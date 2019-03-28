from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.template.util import logger as log_formater
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.herhrl.normalizer import Normalizer
from baselines.herhrl.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.template.policy import Policy
from baselines.herhrl.hrl_policy import HRL_Policy
from baselines.herhrl.ddpg_her_hrl_policy import DDPG_HER_HRL_POLICY
from baselines.herhrl.pddl_policy import PDDL_POLICY
import math

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class MIX_PDDL_HRL_POLICY(DDPG_HER_HRL_POLICY, PDDL_POLICY):
    # Putting DPG_HER_HRL_POLICY before PDDL_POLICY in inheritance assures that the functions in that class have precedence over those in PDDL_POLICY (see https://stackoverflow.com/questions/21657822/python-and-order-of-methods-in-multiple-inheritance)
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
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
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        DDPG_HER_HRL_POLICY.__init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=reuse, **kwargs)

        self.count_pddl = 0.0
        self.count_hrl = 0.0


    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False, exploit=True, success_rate=1.):
        # TODO: Make these two parameters parameterizable.
        p_threshold = 0.2
        p_steepness = 2.0

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def sigm_prob(x, p_threshold, steepness):
            # Verify using https://www.desmos.com/calculator 1/(1+\exp(-(2\cdot\left((x+0.5-0.3)\cdot12-6)))\right)
            offset_x = x - p_threshold
            scaled_x = offset_x * 12
            p = sigmoid(steepness * scaled_x)
            return p

        p_ddpg = sigm_prob(success_rate, p_threshold, p_steepness)
        rnd = np.random.uniform()

        if rnd > p_ddpg:

            u, q = PDDL_POLICY.get_actions(self, o, ag, g)
            self.count_pddl += 1
        else:
            u, q = DDPG_HER_HRL_POLICY.get_actions(self, o, ag, g, noise_eps=noise_eps, random_eps=random_eps, use_target_net=use_target_net,
                    compute_Q=compute_Q, exploit=exploit)
            self.count_hrl += 1

        p_ddpg = sigm_prob(success_rate, p_threshold, p_steepness)

        return u, q

    def logs(self, prefix='policy'):
        logs = []
        logs += [('ddpg/total', float(self.count_hrl / (self.count_hrl + self.count_pddl)))]
        logs += [('pddl/total', float(self.count_pddl / (self.count_hrl + self.count_pddl)))]
        logs = log_formater(logs, prefix + "_{}".format(self.h_level))
        logs += DDPG_HER_HRL_POLICY.logs(self, prefix=prefix)
        self.reset_counters()
        return logs

    def reset_counters(self):
        self.count_hrl = 0.0
        self.count_pddl = 0.0