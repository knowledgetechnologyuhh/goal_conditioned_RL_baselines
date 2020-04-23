import tensorflow as tf
import itertools
import numpy as np

def flatten_mixed_np_array(a):
    semi_flat = list(itertools.chain(*a))
    flat = []
    for item in semi_flat:
        if type(item) == np.float32:
            flat.append([item])
        else:
            flat.append(item)
    flat = list(itertools.chain(*flat))
    return flat

def layer(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]

    if is_output:
        weight_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
        bias_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    else:
        # 1/sqrt(f)
        fan_in_init = 1 / num_prev_neurons ** 0.5
        weight_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)
        bias_init = tf.random_uniform_initializer(minval=-fan_in_init, maxval=fan_in_init)

    weights = tf.get_variable("weights", shape, initializer=weight_init)
    biases = tf.get_variable("biases", [num_next_neurons], initializer=bias_init)

    dot = tf.matmul(input_layer, weights) + biases

    if is_output:
        return dot

    relu = tf.nn.relu(dot)
    return relu

class BasicEnvWrapper(object):

    def __init__(self, env, n_layers, time_scale, input_dims ):
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
        self.action_bounds = np.ones(self.action_dim) # np.array([max_u] * self.action_dim)
        self.action_offset = np.zeros(self.action_dim) # np.zeros((len(self.action_bounds)))
        self.max_actions = self.time_scale**(self.n_layers)

    def set_subgoal_props(self):
        """ use wtm internal methods to specify subgoal properties"""
        self.subgoal_dim = self.end_goal_dim
        scale, offset = self.get_scale_and_offset_for_normalized_subgoal()
        self.subgoal_bounds = np.stack((scale, scale), axis=1)
        self.subgoal_bounds[:2, 0] *= -1.0
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2

        self.subgoal_bounds_offset = offset
        print('dims: action = {}, subgoal = {}, end_goal = {}'.format(self.action_dim, self.subgoal_dim, self.end_goal_dim))
        print('subgoal_bounds: symmetric {}, offset {}'.format(self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))

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

        if self.visualize:
            self.render()
            if self.graph:
                for l in self.agent.layers:
                    if self.agent.model_based:
                        curi = np.mean(l.curiosity) if l.curiosity else 0.0
                        self.add_graph_values('curiosity_layer_{}'.format(l.layer_number), np.array([curi]) ,self.wrapped_env.step_ctr, reset=reset)
                    else:
                        q_val = np.mean(l.q_values) if l.q_values else 0.0
                        self.add_graph_values('q_layer_{}'.format(l.layer_number), np.array([q_val]) ,self.wrapped_env.step_ctr, reset=reset)

        return self._get_obs()['observation']


class AntWrapper(BasicEnvWrapper):

    def __init__(self, env, n_layers, time_scale, input_dims):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims)

        self.subgoal_dim = len(self.subgoal_bounds)
        self.subgoal_bounds_symmetric = np.zeros(self.subgoal_dim)
        self.subgoal_bounds_offset = np.zeros(self.subgoal_dim)
        for i in range(self.subgoal_dim):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        self.project_state_to_end_goal = lambda state : self.wrapped_env._obs2goal(state)
        self.project_state_to_sub_goal = lambda state : self.wrapped_env._obs2subgoal(state)

    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)


class BlockWrapper(BasicEnvWrapper):
    def __init__(self, env, n_layers, time_scale, input_dims):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims)

        self.set_subgoal_props()
        self.project_state_to_sub_goal = lambda state : self.wrapped_env._obs2goal(state)
        self.project_state_to_end_goal = lambda state : self.wrapped_env._obs2goal(state)
        self.sub_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)
        self.end_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)

    def display_end_goal(self, end_goal):
        pass

    def display_subgoals(self, subgoals):
        # TODO: Block environments only works for one subgoal
        self.wrapped_env.goal = subgoals[0]

