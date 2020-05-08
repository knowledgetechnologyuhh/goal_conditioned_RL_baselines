import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from baselines.chac.utils import Base, hidden_init


class Actor(Base):

    def __init__(self, env, layer_number, n_layers,
            learning_rate=0.001, hidden_size=64):
        super(Actor, self).__init__()

        self.actor_name = 'actor_' + str(layer_number)
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine range of actor network outputs.
        # This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = torch.FloatTensor(env.action_bounds).unsqueeze(0).to(self.device)
            self.action_offset = torch.FloatTensor(env.action_offset).unsqueeze(0).to(self.device)
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = torch.FloatTensor(env.subgoal_bounds_symmetric).unsqueeze(0).to(self.device)
            self.action_offset = torch.FloatTensor(env.subgoal_bounds_offset).unsqueeze(0).to(self.device)

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
        x = torch.cat([ state, goal ], dim=1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return torch.tanh(self.fc4(h3)) * self.action_space_bounds + self.action_offset

    def update(self, actor_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
