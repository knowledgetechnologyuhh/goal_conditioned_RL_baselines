import numpy as np
from baselines.template.util import store_args, logger
from baselines.template.policy import Policy
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines.model_based.replay_buffer import ReplayBuffer


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class MBPolicy(Policy):
    @store_args
    def __init__(self, input_dims, T, rollout_batch_size, **kwargs):
        """ Just a random dummy. Does not learn anything
        """
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=False)

        # Configure the replay buffer.
        input_shapes = dims_to_shapes(self.input_dims)
        buffer_shapes = {key: (self.T if key != 'o' else self.T + 1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T + 1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def get_actions(self, o, ag, g, policy_action_params=None):
        # This is important for the rollout (Achieved through policy). DUMMY RETURN ZEROS
        EMPTY = 0
        u = np.random.randn(o.size // self.dimo, self.dimu)
        return u, EMPTY

    def store_episode(self, episode_batch, update_stats=True):
        print("Storing episode")
        pass

    def get_current_buffer_size(self):
        print("Getting current buffer size...")
        pass

    def sample_batch(self):
        print("Sampling batch")
        pass

    def stage_batch(self, batch=None):
        print("Staging batch")
        pass

    def train(self, stage=True):
        print("Training")
        pass

    def clear_buffer(self):
        print("Clearing buffer")
        pass

    def logs(self, prefix=''):
        logs = []
        logs += [('stats/some_stat_value', 0)]
        return logger(logs, prefix)

