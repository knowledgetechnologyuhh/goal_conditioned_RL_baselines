import torch


class ExperienceBuffer:
    def __init__(self, max_buffer_size, batch_size, state_dim, action_dim, goal_dim):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size

        self._states = torch.zeros((self.max_buffer_size, state_dim) ,dtype=torch.float32)
        self._actions = torch.zeros((self.max_buffer_size, action_dim), dtype=torch.float32)
        self._rewards = torch.zeros(self.max_buffer_size, dtype=torch.float32)
        self._new_states = torch.zeros((self.max_buffer_size, state_dim), dtype=torch.float32)
        self._goals = torch.zeros((self.max_buffer_size, goal_dim), dtype=torch.float32)
        self._is_terminals = torch.zeros(self.max_buffer_size, dtype=torch.int)

        self._size = 0
        self._cursor = 0

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, info\')'
        assert type(experience[5]) == bool

        # Enter experience at current cursor
        self._states[self._cursor] = experience[0] if isinstance(experience[0], torch.Tensor) else torch.tensor(experience[0])
        self._actions[self._cursor] = experience[1] if isinstance(experience[1], torch.Tensor) else torch.tensor(experience[1])
        self._rewards[self._cursor] = experience[2] if isinstance(experience[2], torch.Tensor) else torch.tensor(experience[2])
        self._new_states[self._cursor] = experience[3] if isinstance(experience[3], torch.Tensor) else torch.tensor(experience[3])
        self._goals[self._cursor] = experience[4] if isinstance(experience[4], torch.Tensor) else torch.tensor(experience[4])
        self._is_terminals[self._cursor] = experience[5] if isinstance(experience[5], torch.Tensor) else torch.tensor(experience[5])

        self._cursor += 1

        # Increase size until max size is reached
        if self._size < self.max_buffer_size:
            self._size += 1

        # When cursor reaches end, restart at beginning, overwriting oldest entries first
        if self._cursor == self.max_buffer_size:
            self._cursor = 0

    def get_batch(self):
        import numpy as np
        dist = np.random.randint(0, high=self._size, size=min(self._size, self.batch_size))
        #  dist = torch.randint(0, high=self._size, size=min(self._size, self.batch_size))

        # numpy indexing with arrays
        states = self._states[dist]
        actions = self._actions[dist]
        rewards = self._rewards[dist].unsqueeze(1)
        new_states = self._new_states[dist]
        goals = self._goals[dist]
        is_terminals = self._is_terminals[dist].unsqueeze(1)

        return states, actions, rewards, new_states, goals, is_terminals

    @property
    def size(self):
        return self._size
