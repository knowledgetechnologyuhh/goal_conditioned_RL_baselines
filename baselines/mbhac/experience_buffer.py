import numpy as np


class ExperienceBuffer:
    def __init__(self, max_buffer_size, batch_size, state_dim, action_dim, goal_dim):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size

        self._states = np.zeros(dtype=np.float, shape=(self.max_buffer_size, state_dim))
        self._actions = np.zeros(dtype=np.float, shape=(self.max_buffer_size, action_dim))
        self._rewards = np.zeros(dtype=np.float, shape=(self.max_buffer_size,))
        self._new_states = np.zeros(dtype=np.float, shape=(self.max_buffer_size, state_dim))
        self._goals = np.zeros(dtype=np.float, shape=(self.max_buffer_size, goal_dim))
        self._is_terminals = np.zeros(dtype=np.int, shape=(self.max_buffer_size,))

        self._size = 0
        self._cursor = 0

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, info\')'
        assert type(experience[5]) == bool

        # Enter experience at current cursor
        self._states[self._cursor] = experience[0]
        self._actions[self._cursor] = experience[1]
        self._rewards[self._cursor] = experience[2]
        self._new_states[self._cursor] = experience[3]
        self._goals[self._cursor] = experience[4]
        self._is_terminals[self._cursor] = int(experience[5])

        self._cursor += 1

        # Increase size until max size is reached
        if self._size < self.max_buffer_size:
            self._size += 1

        # When cursor reaches end, restart at beginning, overwriting oldest entries first
        if self._cursor == self.max_buffer_size:
            self._cursor = 0

    def get_batch(self):
        dist = np.random.randint(0, high=self._size, size=min(self._size, self.batch_size))

        # numpy indexing with arrays
        states = self._states[dist]
        actions = self._actions[dist]
        rewards = self._rewards[dist]
        new_states = self._new_states[dist]
        goals = self._goals[dist]
        is_terminals = self._is_terminals[dist]

        return states, actions, rewards, new_states, goals, is_terminals

    @property
    def size(self):
        return self._size
