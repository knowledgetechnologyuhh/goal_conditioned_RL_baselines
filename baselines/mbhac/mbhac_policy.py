from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.mbhac.normalizer import Normalizer
from baselines.mbhac.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.template.policy import Policy
from baselines.mbhac.layer import Layer



class MBHACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        self.sess = tf.Session()
        self.n_layers = 2

        self.layers = []

        for i in range(self.n_layers):

            # subgoal action for higher policy
            if i > 0:
                input_dims['u'] = input_dims['o']

            layer = Layer(input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size, Q_lr, pi_lr, norm_eps,
                    norm_clip, max_u, action_l2, clip_obs, scope, T, rollout_batch_size, subtract_goals, relative_goals,
                    clip_pos_returns, clip_return, sample_transitions, gamma, self.sess, i, **kwargs)

            self.layers.append(layer)

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False, exploit=True):
        action_stack = [self.layers[i].get_actions(o, ag, g, noise_eps=0., random_eps=0., use_target_net=use_target_net, compute_Q=compute_Q, exploit=exploit)
                for i in range(self.n_layers)]
        # return primitive action of lowest layer
        return action_stack[-1]

    def store_episode(self, episode_batch, update_stats=True):
        for i in range(self.n_layers):
            self.layers[i].store_episode(episode_batch, update_stats=update_stats)

    def train(self, stage=True):
        return [[self.layers[i].train(stage)] for i in range(self.n_layers)]

    def update_target_net(self):
        for i in range(self.n_layers):
            self.layers[0].update_target_net_op

