import numpy as np

class ExperienceBuffer():

    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, info\')'
        assert type(experience[5]) == bool

        self.experiences.append(experience)

        while self.size and self.size >= self.max_buffer_size:
            self.experiences.pop(0)

    def get_batch(self):
        states, actions, rewards, new_states, goals, is_terminals = [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=min(self.size, self.batch_size))

        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            goals.append(self.experiences[i][4])
            is_terminals.append(self.experiences[i][5])

        return states, actions, rewards, new_states, goals, is_terminals

    @property
    def size(self):
        return len(self.experiences)
