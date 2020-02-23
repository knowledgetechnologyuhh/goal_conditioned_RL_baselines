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


# Below function prints out options and environment specified by user
def print_summary(FLAGS,env):

    print("\n- - - - - - - - - - -")
    print("Task Summary: ","\n")
    print("Environment: ", env.name)
    print("Number of Layers: ", FLAGS.layers)
    print("Time Limit per Layer: ", FLAGS.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", FLAGS.retrain)
    print("Test: ", FLAGS.test)
    print("Visualize: ", FLAGS.show)
    print("- - - - - - - - - - -", "\n\n")


# Below function ensures environment configurations were properly entered
def check_validity(model_name, goal_space_test, goal_space_train, goal_thresholds, initial_state_space, subgoal_bounds, subgoal_thresholds, max_actions, timesteps_per_action):

    # Ensure model file is an ".xml" file
    assert model_name[-4:] == ".xml", "Mujoco model must be an \".xml\" file"

    # Ensure upper bounds of range is >= lower bound of range
    if goal_space_train is not None:
        for i in range(len(goal_space_train)):
            assert goal_space_train[i][1] >= goal_space_train[i][0], "In the training goal space, upper bound must be >= lower bound"

    # if goal_space_test is not None:
        for i in range(len(goal_space_test)):
            assert goal_space_test[i][1] >= goal_space_test[i][0], "In the training goal space, upper bound must be >= lower bound"

    for i in range(len(initial_state_space)):
        assert initial_state_space[i][1] >= initial_state_space[i][0], "In initial state space, upper bound must be >= lower bound"

    for i in range(len(subgoal_bounds)):
        assert subgoal_bounds[i][1] >= subgoal_bounds[i][0], "In subgoal space, upper bound must be >= lower bound"

    # Make sure end goal spaces and thresholds have same first dimension
    if goal_space_train is not None and goal_space_test is not None:
        assert len(goal_space_train) == len(goal_space_test) == len(goal_thresholds), "End goal space and thresholds must have same first dimension"

    # Make sure suboal spaces and thresholds have same dimensions
    assert len(subgoal_bounds) == len(subgoal_thresholds), "Subgoal space and thresholds must have same first dimension"
    # Ensure max action and timesteps_per_action are postive integers
    assert max_actions > 0, "Max actions should be a positive integer"

    assert timesteps_per_action > 0, "Timesteps per action should be a positive integer"

class EnvWrapper(object):
    def __init__(self, env, FLAGS, input_dims):
        self.wrapped_env = env

        # design_agent_and_env
        FLAGS.layers = 2
        if FLAGS.time_scale == 0:
            # Enter max sequence length in which each policy will specialize
            FLAGS.time_scale = 30

        self.FLAGS = FLAGS
        self.visualize = False

        self.state_dim = len(env.sim.data.qpos) + len(env.sim.data.qvel)
        self.action_dim = len(self.sim.model.actuator_ctrlrange)
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1]
        self.action_offset = np.zeros((len(self.action_bounds)))
        self.end_goal_dim = len(self.goal_space_test)
        self.subgoal_dim = len(self.subgoal_bounds)
        print('dims: action = {}, subgoal = {}, end_goal = {}'.format(self.action_dim, self.subgoal_dim, self.end_goal_dim))

        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))
        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        print('subgoal_bounds: symmetric {}, offset {}'.format(self.subgoal_bounds_symmetric, self.subgoal_bounds_offset))

        max_actions = 700
        max_actions = FLAGS.time_scale**(FLAGS.layers)
        self.max_actions = max_actions

    def __getattr__(self, attr):
        return self.wrapped_env.__getattribute__(attr)

    def execute_action(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        #  self._set_action(action)
        if self.visualize:
            self.render()

        return self._get_state()

    # TODO: compare with levy's def
    def get_next_goal(self, test):
        end_goal = np.zeros((len(self.goal_space_test)))
        if self.name == "ant_four_rooms.xml":

            # Randomly select one of the four rooms in which the goal will be located
            room_num = np.random.randint(0,4)

            # Pick exact goal location
            end_goal[0] = np.random.uniform(3,6.5)
            end_goal[1] = np.random.uniform(3,6.5)
            end_goal[2] = np.random.uniform(0.45,0.55)

            # If goal should be in top left quadrant
            if room_num == 1:
                end_goal[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif room_num == 2:
                end_goal[0] *= -1
                end_goal[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif room_num == 3:
                end_goal[1] *= -1



        elif not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])

        return end_goal

    #  def reset_sim(self, next_goal=None):
    #      return env.reset()

