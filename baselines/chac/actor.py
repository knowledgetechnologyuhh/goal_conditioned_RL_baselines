import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baselines.chac.utils import Base, hidden_init


class Actor(Base):
    def __init__(self, env, level, n_levels, device, lr=0.001, hidden_size=64):
        super(Actor, self).__init__()

        self.actor_name = 'actor_' + str(level)

        # Determine range of actor network outputs.
        self.action_space_bounds = torch.FloatTensor(env.action_bounds
                if level == 0 else env.subgoal_bounds_symmetric).to(device)
        self.action_offset = torch.FloatTensor(env.action_offset
                if level == 0 else env.subgoal_bounds_offset).to(device)

        # Dimensions of action will depend on layer level
        action_space_size = env.action_dim if level == 0 else env.subgoal_dim

        # Dimensions of goal placeholder will differ depending on layer level
        goal_dim = env.end_goal_dim if level == n_levels - 1 else env.subgoal_dim

        # Network layers
        self.fc1 = nn.Linear(env.state_dim + goal_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_space_size)

        self.actor_optimizer = optim.Adam(self.parameters(), lr)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.bias.data.uniform_(*hidden_init(self.fc3))
        self.fc4.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return torch.tanh(self.fc4(h3)) * self.action_space_bounds + self.action_offset

    def update(self, mu_loss):
        self.actor_optimizer.zero_grad()
        mu_loss.backward()
        flat_grads = torch.cat([param.flatten() for _, param in self.named_parameters()])
        self.actor_optimizer.step()
        return {
            'mu_loss': mu_loss.item(),
            'mu_grads': flat_grads.mean().item(),
            'mu_grads_std': flat_grads.std().item(),
        }
