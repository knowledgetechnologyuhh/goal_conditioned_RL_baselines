import torch


class ExperienceBuffer:
    def __init__(self, max_buffer_size, batch_size, state_dim, action_dim, goal_dim):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._states = torch.empty((self.max_buffer_size, state_dim) ,dtype=torch.float32)
        self._actions = torch.empty((self.max_buffer_size, action_dim), dtype=torch.float32)
        self._rewards = torch.empty(self.max_buffer_size, dtype=torch.float32)
        self._new_states = torch.empty((self.max_buffer_size, state_dim), dtype=torch.float32)
        self._goals = torch.empty((self.max_buffer_size, goal_dim), dtype=torch.float32)
        self._done = torch.empty(self.max_buffer_size, dtype=torch.int)

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
        self._done[self._cursor] = experience[5] if isinstance(experience[5], torch.Tensor) else torch.tensor(experience[5])

        self._cursor += 1

        # Increase size until max size is reached
        if self._size < self.max_buffer_size:
            self._size += 1

        # When cursor reaches end, restart at beginning, overwriting oldest entries first
        if self._cursor == self.max_buffer_size:
            self._cursor = 0

    def get_batch(self):
        dist = torch.randint(0, high=self._size, size=(min(self._size, self.batch_size),))
        states = self._states[dist].to(self.device)
        actions = self._actions[dist].to(self.device)
        rewards = self._rewards[dist].unsqueeze(1).to(self.device)
        new_states = self._new_states[dist].to(self.device)
        goals = self._goals[dist].to(self.device)
        done = self._done[dist].unsqueeze(1).to(self.device)

        return states, actions, rewards, new_states, goals, done

    @property
    def size(self):
        return self._size
