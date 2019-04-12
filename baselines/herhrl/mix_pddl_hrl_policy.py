from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from queue import deque
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

# class FlexiAvgList:
#     def __init__(self, maxlen=1000, n_chunks=4):
#         self.counter = 0
#         self.buffer = []
#         self.this_counter = 0
#         self.maxlen = maxlen
#
#     def add(self, item):
#         self.buffer.append(item)
#         self.counter += 1
#         self.this_counter += 1
#         if self.this_counter > self.maxlen:



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
        # self.p_threshold = kwargs['p_threshold']
        self.p_threshold = {'train': 1.0, 'test': 1.0}
        self.p_steepness = kwargs['p_steepness']
        self.train_min_mix_entries = self.buffer_size // 8
        self.train_min_mix_entries = 2004
        # self.min_mix_entries = 20
        self.succ_rate_history = {'train': deque(maxlen=self.buffer_size // 2),
                                  'test': deque(maxlen=self.buffer_size // 2)}

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False, exploit=True, success_rate=1.):

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def sigm_prob(x, p_threshold, steepness):
            # Verify using https://www.desmos.com/calculator 1/(1+\exp(-(2\cdot\left((x+0.5-0.3)\cdot12-6)))\right)
            offset_x = x - p_threshold
            scaled_x = offset_x * 12
            p = sigmoid(steepness * scaled_x)
            return p
        mode = 'train'
        if exploit:
            mode = 'test'

        self.succ_rate_history[mode].append(np.mean(success_rate))

        # If last 4/4 of history is not 5% better than last 3/4 of history after at least self.min_mix_entries entries, set switch point.
        # The switch point sets the threshold for sigmoidal policy sampling.
        if len(self.succ_rate_history['train']) > self.train_min_mix_entries and self.p_threshold[mode] == 1.0 and len(self.succ_rate_history['test']) > 4:
            frac_n = len(self.succ_rate_history[mode]) // 4
            prev = list(self.succ_rate_history[mode])[-(2*frac_n):-(frac_n)]
            last = list(self.succ_rate_history[mode])[-frac_n:]
            avg_prev = np.mean(prev)
            avg_last = np.mean(last)
            if avg_prev * 1.02 > avg_last and avg_last > 0.01:
                self.p_threshold[mode] = avg_last

        p_ddpg = sigm_prob(success_rate, self.p_threshold[mode], self.p_steepness)
        rnd = np.random.uniform()
        u, q = DDPG_HER_HRL_POLICY.get_actions(self, o, ag, g, noise_eps=noise_eps, random_eps=random_eps,
                                               use_target_net=use_target_net,
                                               compute_Q=compute_Q, exploit=exploit)
        if rnd > p_ddpg:
            scaled_u, q_pddl = PDDL_POLICY.get_actions(self, o, ag, g)
            # This u comes already in scale and with offsets, i.e., not in [-1,1]. So we have to descale it.
            u = self.inverse_scale_and_offset_action(scaled_u)
            if u.shape != g.shape:
                u = np.reshape(u, g.shape)  # this to solve the unbalance issue when rollout_batch_size = 1
            self.count_pddl += 1
        else:
            self.count_hrl += 1

        return u, q

    def logs(self, prefix='policy'):
        logs = []
        logs += [('ddpg_per_total', float(self.count_hrl / (self.count_hrl + self.count_pddl)))]
        logs += [('pddl_per_total', float(self.count_pddl / (self.count_hrl + self.count_pddl)))]
        logs += [('p_threshold_train', float(self.p_threshold['train']))]
        logs += [('p_threshold_test', float(self.p_threshold['test']))]

        logs = log_formater(logs, prefix + "_{}".format(self.h_level))
        logs += DDPG_HER_HRL_POLICY.logs(self, prefix=prefix)
        self.reset_counters()
        return logs

    def reset_counters(self):
        self.count_hrl = 0.0
        self.count_pddl = 0.0