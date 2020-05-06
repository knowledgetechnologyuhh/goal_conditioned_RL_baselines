import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baselines.chac.utils import Base


class Actor(Base):

    def __init__(self, env, batch_size, layer_number, n_layers,
            learning_rate=0.001, hidden_size=64):
        super(Actor, self).__init__()

        self.actor_name = 'actor_' + str(layer_number)
        self.hidden_size = hidden_size

        # Determine range of actor network outputs.
        # This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = torch.FloatTensor(env.action_bounds)
            self.action_offset = torch.FloatTensor(env.action_offset)
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = torch.FloatTensor(env.subgoal_bounds_symmetric)
            self.action_offset = torch.FloatTensor(env.subgoal_bounds_offset)

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == n_layers - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.state_dim = env.state_dim
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_dim + self.goal_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.action_space_size)
        self.actor_optimizer = optim.Adam(self.parameters(), learning_rate)

        # init weights
        self.reset()

    def forward(self, state, goal):
        x = torch.cat([ state, goal ], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return (torch.tanh(self.fc4(x)) * self.action_space_bounds) + self.action_offset

    def update(self, actor_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
