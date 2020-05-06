import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from baselines.chac.utils import Base, mlp

class ForwardModel(Base):

    def __init__(self, env, layer_number, mb_params, err_list_size):

        super(ForwardModel, self).__init__()
        self.model_name = 'model_' + str(layer_number)

        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.hidden_sizes = [int(size) for size in mb_params['hidden_size'].split(',')]
        self.eta = mb_params['eta']


        self.mlp = mlp([env.state_dim + self.action_space_size] + self.hidden_sizes + \
                [env.state_dim], nn.Identity)

        self.fw_optimizer = optim.Adam(self.parameters(),  mb_params['lr'])

        # init weights
        self.reset()

        self.err_list_size = err_list_size
        self.err_list = []

    def forward(self, action, state):
        x = torch.cat((action.float(), state.float()), dim=1)
        return self.mlp(x)

    def normalize_bonus(self, bonus_lst):
        """ Bonus range between -1.0 and 0.0 """
        norm_bonus = (bonus_lst - self.min_err) / (self.max_err - self.min_err)
        return norm_bonus - 1.0

    @torch.no_grad()
    def pred_bonus(self, action, state, s_next):
        s_next_prediction = self(action, state)
        errs = (s_next_prediction - s_next) ** 2
        err = errs.mean(axis=1)

        if len(self.err_list) < self.err_list_size and err.size:
            self.err_list += err.tolist()
            # update bounds for normalization
            self.min_err = np.min(self.err_list)
            self.max_err = np.max(self.err_list)

        return self.normalize_bonus(err)

    def update(self, states, actions, new_states):
        self.train()
        self.fw_optimizer.zero_grad()
        state_prediction = self(actions, states)
        loss = F.mse_loss(state_prediction, new_states)
        loss.backward()
        self.fw_optimizer.step()

        return loss.item()
