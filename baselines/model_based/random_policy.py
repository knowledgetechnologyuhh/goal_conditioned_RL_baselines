import numpy as np
from baselines.template.util import store_args, logger
from baselines.template.policy import Policy

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class RandomPolicy(Policy):
    @store_args
    def __init__(self, input_dims, T, rollout_batch_size, **kwargs):
        """ Just a random dummy. Does not learn anything
        """
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

    def get_actions(self, o, ag, g, policy_action_params=None):
        # This is important for the rollout (Achieved through policy). DUMMY RETURN ZEROS
        EMPTY = 0
        u = np.random.randn(o.size // self.dimo, self.dimu)
        return u, EMPTY

    def store_episode(self, episode_batch, update_stats=True):
        pass

    def get_current_buffer_size(self):
        pass

    def sample_batch(self):
        pass

    def stage_batch(self, batch=None):
        pass

    def train(self, stage=True):
        pass

    def clear_buffer(self):
        pass

    def logs(self, prefix=''):
        logs = []
        logs += [('stats/some_stat_value', 0)]
        return logger(logs, prefix)

