import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baselines.chac.utils import Base


class Critic(Base):

    def __init__(self, env, layer_number, n_layers, time_scale,
            learning_rate=0.001, gamma=0.98, tau=0.05, hidden_size=64):

        super(Critic, self).__init__()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.hidden_size = hidden_size
        self.q_limit = -time_scale
        self.state_dim = env.state_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == n_layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -torch.tensor([self.q_limit/self.q_init - 1]).log().view(1, -1)

        self.fc1 = nn.Linear(self.state_dim + action_dim + self.goal_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

        self.critic_optimizer = optim.Adam(self.parameters(), learning_rate)
        self.mse_loss = nn.MSELoss()

        # init weights
        self.reset()

    def forward(self, state, goal, action):
        x = torch.cat([ state, action, goal ], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x) + self.q_offset) * self.q_limit

    def update(self, old_states, old_actions, rewards, new_states, goals, new_actions, is_terminals):
        next_q = self(new_states, goals, new_actions)
        target_q = rewards + (self.gamma * next_q * (1. - is_terminals)).detach()
        current_q = self(old_states, goals, old_actions)

        #  critic_loss = self.mse_loss(current_q, target_q)
        critic_loss = F.smooth_l1_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
                "critic_loss" : critic_loss.item(),
                'target_q': target_q.mean().item(),
                'next_q': next_q.mean().item(),
                'current_q': current_q.mean().item()
                }
