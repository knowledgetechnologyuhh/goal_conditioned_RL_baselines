import threading
import random
import numpy as np


class ModelReplayBuffer:
    def __init__(self, buffer_shapes, size):
        """Creates a replay buffer to train the model.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size (int): the size of the buffer, measured in rollouts
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size

        self.buffer = []

        # memory management
        # self.current_size = 0
        # self.n_transitions_stored = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return len(self.buffer) == self.size

    def sample(self, batch_size):
        """Returns a list of episodes
        """
        # episodes = []

        with self.lock:
            assert len(self.buffer) > 0
            episodes = np.random.choice(self.buffer, batch_size)

        return episodes

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        with self.lock:
            for ep in episode_batch:
                self.buffer.append(ep)
            if len(self.buffer) > self.size:
                rnd_idx = random.randint(0, len(self.buffer))
                self.buffer.remove(self.buffer[rnd_idx])

    # def get_current_episode_size(self):
    #     with self.lock:
    #         return self.current_size
    #
    # def get_current_size(self):
    #     with self.lock:
    #         return self.current_size * self.T
    #
    # def get_transitions_stored(self):
    #     with self.lock:
    #         return self.n_transitions_stored
    #
    # def clear_buffer(self):
    #     with self.lock:
    #         self.current_size = 0
    #
    # def _get_storage_idx(self, inc=None):
    #     inc = inc or 1   # size increment
    #     assert inc <= self.size, "Batch committed to replay is too large!"
    #     # go consecutively until you hit the end, and then go randomly.
    #     if self.current_size+inc <= self.size:
    #         idx = np.arange(self.current_size, self.current_size+inc)
    #     elif self.current_size < self.size:
    #         overflow = inc - (self.size - self.current_size)
    #         idx_a = np.arange(self.current_size, self.size)
    #         idx_b = np.random.randint(0, self.current_size, overflow)
    #         idx = np.concatenate([idx_a, idx_b])
    #     else:
    #         idx = np.random.randint(0, self.size, inc)
    #
    #     # update replay size
    #     self.current_size = min(self.size, self.current_size+inc)
    #
    #     if inc == 1:
    #         idx = idx[0]
    #     return idx
