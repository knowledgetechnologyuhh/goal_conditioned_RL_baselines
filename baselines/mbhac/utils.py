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

    def __init__(self, env, n_layers, time_scale, input_dims, max_u, agent):
        self.wrapped_env = env
        self.n_layers = n_layers
        self.time_scale = time_scale
        self.visualize = False
        self.graph = self.visualize
        self.agent = agent

        self.state_dim = input_dims['o']
        self.action_dim = input_dims['u']
        self.end_goal_dim = input_dims['g']
        self.action_bounds =  np.array([max_u] * self.action_dim)
        self.action_offset = np.zeros((len(self.action_bounds)))
        self.max_actions = self.time_scale**(self.n_layers)

    def set_sym_bounds(self):
        print('dims: action = {}, subgoal = {}, end_goal = {}'.format(self.action_dim, self.subgoal_dim, self.end_goal_dim))
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))
        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]
        print('subgoal_bounds: symmetric {}, offset {}'.format(self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))


    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)

    def execute_action(self, action):
        self._set_action(action)
        self.sim.step()
        self._step_callback()

        if self.visualize:
            self.render()
            if self.graph:
                reset = self.step_ctr == 0
                for l in self.agent.layers:
                    if self.agent.model_based:
                        #  curi = np.mean(l.curiosity) if l.curiosity else 0.0
                        curi = l.curiosity[-1] if l.curiosity else 0.0
                        self.add_graph_values('curiosity_layer_{}'.format(l.layer_number), np.array([curi]) ,self.step_ctr, reset=reset)
                    #  TODO: Show other metric #

        # TODO: _get_state calls _obs2goal. For layers > 0 we need to call
        #       _get_obs2subgoal to do it like Levy did it.
        return self._get_state()


class AntWrapper(BasicEnvWrapper):

    def __init__(self, env, n_layers, time_scale, input_dims, max_u, agent):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims, max_u, agent)

        assert len(env.sim.data.qpos) + len(env.sim.data.qvel) == self.state_dim
        assert len(self.sim.model.actuator_ctrlrange) == self.action_dim
        assert (self.sim.model.actuator_ctrlrange[:,1] == self.action_bounds).all()
        assert len(self.goal_space_test) == self.end_goal_dim
        # If you like the subgoal space to be equal to the actual space
        #  self.subgoal_bounds = env.initial_state_space[:5]
        self.subgoal_dim = len(self.subgoal_bounds)
        self.project_state_to_end_goal = lambda state : self.wrapped_env._obs2goal(state)
        self.project_state_to_sub_goal = lambda state : self.wrapped_env._obs2subgoal(state)

        self.set_sym_bounds()

    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)


class BlockWrapper(BasicEnvWrapper):
    def __init__(self, env, n_layers, time_scale, input_dims, max_u, agent):
        BasicEnvWrapper.__init__(self, env, n_layers, time_scale, input_dims, max_u, agent)

        scale, offset = self.get_scale_and_offset_for_normalized_subgoal()
        self.subgoal_bounds = np.stack((scale, scale), axis=1)
        self.subgoal_bounds[:2, 0] *= -1.0

        #  self.subgoal_bounds = np.array([[-self.obj_range,self.obj_range],[-self.obj_range,self.obj_range]])
        #  if self.end_goal_dim == 3:
            #  self.subgoal_bounds = np.concatenate((self.subgoal_bounds,
            #  [[env.table_height, env.table_height + self.obj_height * self.n_objects]]))

        if self.end_goal_dim == 6:
            #  TODO: Find real max_velo
            max_velo = 3
            self.subgoal_bounds = np.concatenate((self.subgoal_bounds,
                [[-max_velo, max_velo], [-max_velo, max_velo], [-max_velo, max_velo]]))

        self.subgoal_dim = self.end_goal_dim

        self.sub_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)
        self.end_goal_thresholds = np.array([self.distance_threshold] * self.end_goal_dim)

        self.project_state_to_sub_goal = lambda state : self.wrapped_env._obs2goal(state)
        self.project_state_to_end_goal = lambda state : self.wrapped_env._obs2goal(state)

        self.set_sym_bounds()
        self.subgoal_bounds_offset = offset

    def display_end_goal(self, end_goal):
        self.wrapped_env.final_goal = end_goal
        # self.final_goal = end_goal
        pass

    def display_subgoals(self, subgoals):
        # TODO: fix for multiple layers
        # self.wrapped_env.goal = subgoals[0]
        # self.goal = subgoals[1]
        self.wrapped_env.goal_hierarchy = reversed(subgoals)
        # TODO: This is how it is done in herhrl: self.first_env.env.goal_hierarchy[self.h_level] = self.g.copy()
        pass

    def _get_state(self):
        return self._get_obs()['observation']

    def _reset_sim(self, next_goal):
        self.wrapped_env._reset_sim()
        return self._get_obs()
