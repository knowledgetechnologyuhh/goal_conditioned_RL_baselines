import threading
import random
import numpy as np


class ModelReplayBuffer:
    def __init__(self, buffer_shapes, size):
        """Creates a replay buffer to train the model.

        Args:
            size (int): the size of the buffer, measured in rollouts
            sample (function): a function that samples episodes from the replay buffer
        """
        self.size = size

        self.buffers = {}
        for key, shape in buffer_shapes.items():
            n_steps = shape[1]
            dim = shape[2]
            self.buffers[key] = np.empty([self.size, n_steps, dim])

        # memory management
        self.current_size = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        # buffers = {}
        batch = {}
        for key in self.buffers.keys():
            batch[key] = np.zeros((batch_size, self.buffers[key].shape[1], self.buffers[key].shape[2]))
        if self.current_size == 0:
            return batch

        with self.lock:
            assert self.current_size > 0
            replace = self.current_size < self.size
            idxs = np.random.choice(self.current_size, batch_size, replace=replace)
            for b_idx,idx in enumerate(idxs):
                # episode = {}
                for key in self.buffers.keys():
                    batch[key][b_idx] = self.buffers[key][idx]
                    # episode[key] = self.buffers[key][idx]
                # ep_batch.append(episode)

        # ep_batch = np.array(ep_batch)
        return batch

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        with self.lock:
            for ep in episode_batch:
                if self.current_size < self.size:
                    ins_idx = self.current_size
                    self.current_size += 1
                else:
                    ins_idx = random.randint(0, self.size-1)

                for key in ep:
                    self.buffers[key][ins_idx] = np.array(ep[key])


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
