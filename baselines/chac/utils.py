import numpy as np
import torch.nn as nn
import gym
from baselines import logger


class Base(nn.Module):
    reset_type = 'xavier'

    def _init_weights(self, m):
        """Recursive apply initializations to components of module"""

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


class EnvWrapper(object):
    def __init__(self, env_name, env, time_scales, input_dims):
        self.name = env_name
        self.wrapped_env = env
        self.visualize = False
        self.graph = self.visualize
        self.agent = None
        self.state_dim = input_dims['o']
        self.action_dim = input_dims['u']
        self.end_goal_dim = input_dims['g']
        self.action_bounds = np.ones(self.action_dim)
        self.action_offset = np.zeros(self.action_dim)
        # maximum number of actions as product of steps per level
        self.max_actions = np.prod(time_scales)

        if hasattr(env, 'subgoal_bounds'):
            # some enviroments have a predefined subgoal space
            self.subgoal_dim = len(env.subgoal_bounds)
            self.subgoal_bounds_symmetric = np.zeros(self.subgoal_dim)
            self.subgoal_bounds_offset = np.zeros(self.subgoal_dim)
            for i in range(self.subgoal_dim):
                self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2
                self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        else:
            # otherwise we assume the end goal space to be equal to the sub goal space
            self.subgoal_dim = self.end_goal_dim
            scale, offset = self.get_scale_and_offset_for_normalized_subgoal()
            self.subgoal_bounds = np.stack((scale, scale), axis=1)
            self.subgoal_bounds[:2, 0] *= -1.0
            self.subgoal_bounds_offset = offset
            self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
            for i in range(len(self.subgoal_bounds)):
                self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]) / 2

            self.sub_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)
            self.end_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)

        logger.info('dims: action = {}, subgoal = {}, end_goal = {}'.format(
            self.action_dim, self.subgoal_dim, self.end_goal_dim))
        logger.info('subgoal_bounds: symmetric {}, offset {}'.format(
            self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))

        self.project_state_to_end_goal = lambda state: env._obs2goal(state)
        if hasattr(env, '_obs2subgoal'):
            # use predefined method of environment
            self.project_state_to_sub_goal = lambda state: env._obs2subgoal(state)
        else:
            if hasattr(env, 'project_state_to_sub_goal'):
                # wrap lambda to only input state
                self.project_state_to_sub_goal = lambda state: env.project_state_to_sub_goal(env.sim, state)
            else:
                # project to end goal space
                self.project_state_to_sub_goal = lambda state: env._obs2goal(state)

    def display_end_goal(self, endgoal):
        if hasattr(self.wrapped_env, 'display_end_goal'):
            self.wrapped_env.display_end_goal(endgoal)
        else:
            return endgoal

    def display_subgoals(self, subgoals):
        if hasattr(self.wrapped_env, 'display_subgoals'):
            self.wrapped_env.display_subgoals(subgoals)
        else:
            # Block environments only works for one subgoal
            if 'Block' in self.name:
                self.wrapped_env.goal = subgoals[0]
            else:
                pass

    def __getattr__(self, attr):
        try:
            return self.wrapped_env.__getattribute__(attr)
        except KeyError:
            raise AttributeError(attr)

    def execute_action(self, action):
        if self.graph:
            reset = self.wrapped_env.step_ctr == 0
        self._set_action(action)
        self.sim.step()
        self._step_callback()

        if self.visualize:
            self.render()
            if self.graph and self.agent:
                for l in self.agent.layers:
                    if self.agent.fw:
                        surprise = l.surprise_history[-1] if l.surprise_history else 0.0
                        self.add_graph_values('surprise_layer_{}'.format(l.level),
                                              np.array([surprise]),
                                              self.wrapped_env.step_ctr,
                                              reset=reset)
                    else:
                        q_val = l.q_values[-1] if l.q_values else 0.0
                        self.add_graph_values('q_layer_{}'.format(l.level),
                                              np.array([q_val]),
                                              self.wrapped_env.step_ctr,
                                              reset=reset)
                    if reset:
                        l.q_values.clear()
                        l.surprise_history.clear()

        return self._get_obs()['observation']


def prepare_env(env_name, time_scale, input_dims):
    return EnvWrapper(env_name, gym.make(env_name).env, time_scale, input_dims)
