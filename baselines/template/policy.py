from collections import OrderedDict

import numpy as np
from baselines.template.util import store_args, logger

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class Policy(object):
    @store_args
    def __init__(self, input_dims, T, rollout_batch_size, **kwargs):
        """The Abstract class for any policy.

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per agent
        """
        self.input_dims = input_dims
        self.T = T
        self.rollout_batch_size = rollout_batch_size

        self.input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *self.input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes

    def get_actions(self, o, ag, g, policy_action_params=None):
        raise NotImplementedError

    def store_episode(self, episode_batch, update_stats=True):
        raise NotImplementedError

    def get_current_buffer_size(self):
        raise NotImplementedError

    def sample_batch(self):
        raise NotImplementedError

    def stage_batch(self, batch=None):
        raise NotImplementedError

    def train(self, stage=True):
        raise NotImplementedError

    def clear_buffer(self):
        raise NotImplementedError

    def logs(self, prefix=''):
        raise NotImplementedError
