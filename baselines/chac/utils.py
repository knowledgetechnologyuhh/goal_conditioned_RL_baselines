import numpy as np
import torch.nn as nn
import gym
from baselines import logger

class Base(nn.Module):
    reset_type = 'xavier'

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            if hasattr(m, 'weight'):

                if self.reset_type == "xavier":
                    nn.init.xavier_uniform_(m.weight.data)
                elif self.reset_type == "zeros":
                    nn.init.constant_(m.weight.data, 0.)
                    nn.init.constant_(m.weight.data, 0.)
                else:
                    raise ValueError("Unknown reset type")

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.uniform_(m.bias.data, *hidden_init(m))

    def reset(self):
        self.apply(self._init_weights)


def hidden_init(layer):
    fan_in = layer.weight.data.size(-1)
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class BasicEnvWrapper(object):
    def __init__(self, env, n_layers, time_scale, input_dims):
        self.wrapped_env = env
        self.n_layers = n_layers
        self.time_scale = time_scale
        self.visualize = False
        self.graph = self.visualize
        # set in config
        self.agent = None
        self.state_dim = input_dims['o']
        self.action_dim = input_dims['u']
        self.end_goal_dim = input_dims['g']
        self.action_bounds = np.ones(self.action_dim)
        self.action_offset = np.zeros(self.action_dim)
        self.max_actions = self.time_scale**(self.n_layers)

    def set_subgoal_props(self):
        """ use wtm internal methods to specify subgoal properties"""
        self.subgoal_dim = self.end_goal_dim
        scale, offset = self.get_scale_and_offset_for_normalized_subgoal()
        self.subgoal_bounds = np.stack((scale, scale), axis=1)
        self.subgoal_bounds[:2, 0] *= -1.0
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] -
                                                self.subgoal_bounds[i][0]) / 2

        self.subgoal_bounds_offset = offset
        logger.info('dims: action = {}, subgoal = {}, end_goal = {}'.format(
            self.action_dim, self.subgoal_dim, self.end_goal_dim))
        logger.info('subgoal_bounds: symmetric {}, offset {}'.format(
            self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))

    def __getattr__(self, attr):
        try:
            return self.wrapped_env.__getattribute__(attr)
        except KeyError:
            raise AttributeError(attr)

    def execute_action(self, action):
        if self.graph: reset = self.wrapped_env.step_ctr == 0
        self._set_action(action)
        self.sim.step()
        self._step_callback()

        #  if self.visualize and self.agent.test_mode:
        if self.visualize:
            self.render()
            if self.graph and self.agent:
                for l in self.agent.layers:
                    if self.agent.fw:
                        curi = np.mean(l.curiosity_hist) if l.curiosity_hist else 0.0
                        self.add_graph_values('curiosity_layer_{}'.format(l.level),
                                              np.array([curi]),
                                              self.wrapped_env.step_ctr,
                                              reset=reset)
                    else:
                        q_val = np.mean(l.q_values) if l.q_values else 0.0
                        self.add_graph_values('q_layer_{}'.format(l.level),
                                              np.array([q_val]),
                                              self.wrapped_env.step_ctr,
                                              reset=reset)

        return self._get_obs()['observation']


class AntWrapper(BasicEnvWrapper):
    def __init__(self, env, n_layers, time_scale, input_dims):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims)

        self.subgoal_dim = len(self.subgoal_bounds)
        self.subgoal_bounds_symmetric = np.zeros(self.subgoal_dim)
        self.subgoal_bounds_offset = np.zeros(self.subgoal_dim)
        for i in range(self.subgoal_dim):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] -
                                                self.subgoal_bounds[i][0]) / 2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        self.project_state_to_end_goal = lambda state: self.wrapped_env._obs2goal(state)
        self.project_state_to_sub_goal = lambda state: self.wrapped_env._obs2subgoal(state)
        logger.info('dims: action = {}, subgoal = {}, end_goal = {}'.format(
            self.action_dim, self.subgoal_dim, self.end_goal_dim))
        logger.info('subgoal_bounds: symmetric {}, offset {}'.format(
            self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))

    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)


class UR5Wrapper(BasicEnvWrapper):
    def __init__(self, env, n_layers, time_scale, input_dims):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims)
        self.subgoal_dim = len(self.subgoal_bounds)
        self.project_state_to_end_goal = lambda state: self.wrapped_env._obs2goal(state)
        self.project_state_to_sub_goal = lambda state: self.wrapped_env.project_state_to_sub_goal(self.wrapped_env.sim, state)

    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)


class BlockWrapper(BasicEnvWrapper):
    def __init__(self, env, n_layers, time_scale, input_dims):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims)

        self.set_subgoal_props()
        self.project_state_to_sub_goal = lambda state: self.wrapped_env._obs2goal(state)
        self.project_state_to_end_goal = lambda state: self.wrapped_env._obs2goal(state)
        self.sub_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)
        self.end_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)

    def display_end_goal(self, end_goal):
        pass

    def display_subgoals(self, subgoals):
        # TODO: Block environments only works for one subgoal
        self.wrapped_env.goal = subgoals[0]


def prepare_env(env_name, n_layers, time_scale, input_dims):
    wrapper_args = (gym.make(env_name).env, n_layers, time_scale, input_dims)
    if 'Ant' in env_name:
        env = AntWrapper(*wrapper_args)
    elif 'UR5' in env_name:
        env = UR5Wrapper(*wrapper_args)
    elif 'Block' in env_name:
        env = BlockWrapper(*wrapper_args)
    elif 'Causal' in env_name:
        env = BlockWrapper(*wrapper_args)
    elif 'Hook' in env_name:
        env = BlockWrapper(*wrapper_args)
    elif 'CopReacher' in env_name:
        env = BlockWrapper(*wrapper_args)

    return env
