import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == n_layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        self.fcs1 = nn.Linear(self.state_dim, hidden_size)
        self.fcs2 = nn.Linear(hidden_size, hidden_size)

        self.fca1 = nn.Linear(action_dim, hidden_size)

        self.fcg1 = nn.Linear(self.goal_dim, hidden_size)
        self.fcg2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size * 3, 1)

        self.critic_optimizer = optim.Adam(self.parameters(), learning_rate)

        # init weights
        self.reset()

    def forward(self, state, goal, action):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state)

        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action)

        if not isinstance(goal, torch.Tensor):
            goal = torch.from_numpy(goal)

        s1 = F.relu(self.fcs1(state.float()))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action.float()))

        g1 = F.relu(self.fcg1(goal.float()))
        g2 = F.relu(self.fcg2(g1))

        x = torch.cat((s2, a1, g2), dim=1)

        x = F.relu(self.fc3(x))
        output = torch.sigmoid(x + self.q_offset) * self.q_limit
        return output

    def update(self, old_states, old_actions, rewards, new_states, goals, new_actions, is_terminals):
        self.train()
        next_q = self(new_states, goals, new_actions)
        target_q = rewards + self.gamma * next_q * (1. - is_terminals)
        self.critic_optimizer.zero_grad()

        current_q = self(old_states, goals, old_actions)

        # Huber loss
        self.loss_val = F.smooth_l1_loss(current_q, target_q)
        self.loss_val.backward()
        self.critic_optimizer.step()

        return {
                "critic_loss" : self.loss_val.item(),
                'target_q': target_q.mean().item(),
                'next_q': next_q.mean().item(),
                'current_q': current_q.mean().item()
                }
