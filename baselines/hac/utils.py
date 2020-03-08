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

def layer_goal_nn(input_layer, num_next_neurons, is_output=False):
    num_prev_neurons = int(input_layer.shape[1])
    shape = [num_prev_neurons, num_next_neurons]


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


class EnvWrapper(object):
    def __init__(self, env, n_layers, time_scale, input_dims, max_u):
        self.wrapped_env = env

        # TODO: get this from click options
        self.n_layers = n_layers
        self.time_scale = time_scale

        if self.time_scale == 0:
            # Enter max sequence length in which each policy will specialize
            self.time_scale = 30

        self.visualize = False

        # TODO: use wtm definitions
        self.state_dim = input_dims['o']
        assert len(env.sim.data.qpos) + len(env.sim.data.qvel) == self.state_dim

        self.action_dim = input_dims['u']
        assert len(self.sim.model.actuator_ctrlrange) == self.action_dim

        self.action_bounds =  np.array([max_u] * self.action_dim)
        assert (self.sim.model.actuator_ctrlrange[:,1] == self.action_bounds).all()

        # n_actions in wtm env
        self.action_offset = np.zeros((len(self.action_bounds)))
        #  print('action offset', self.action_offset)

        self.end_goal_dim = input_dims['g']
        assert len(self.goal_space_test) == self.end_goal_dim

        self.subgoal_dim = len(self.subgoal_bounds)
        #  print('sub goal dim', self.subgoal_dim)

        print('dims: action = {}, subgoal = {}, end_goal = {}'.format(self.action_dim, self.subgoal_dim, self.end_goal_dim))

        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))
        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        print('subgoal_bounds: symmetric {}, offset {}'.format(self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))

        max_actions = self.time_scale**(self.n_layers)
        self.max_actions = max_actions

    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)

    def execute_action(self, action):
        self._set_action(action)
        for _ in range(10):         # TEST
            self.sim.step()
            self._step_callback()

            if self.visualize:
                self.render()

        # TODO: _get_state calls _obs2goal. For layers > 0 we need to call
        #       _get_obs2subgoal to do it like Levy did it.
        return self._get_state()

    # TODO: Check if this version works
    def get_next_goal(self, test):

        if not test and self.goal_space_train is not None:
            return self._sample_goal()

        end_goal = np.zeros((len(self.goal_space_test)))
        assert self.goal_space_test is not None, "Need goal space for testing"
        for i in range(len(self.goal_space_test)):
            end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])
        return end_goal
