from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.util import (
        import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.common.mpi_adam import MpiAdam
from baselines.template.policy import Policy
from baselines.hac.layer import Layer

from baselines.hac.agent import Agent
import baselines.hac.env_designs
from baselines.hac.options import parse_options
import os,sys,inspect
import importlib


class HACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, polyak, batch_size,
            Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
            rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
            sample_transitions, gamma, reuse=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir)

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
        self.FLAGS = parse_options()
        self.FLAGS.mix_train_test = True
        self.FLAGS.retrain = True
        self.FLAGS.show = True

        agent, env = self.init_levy(self.FLAGS)
        wtm_agent, wtm_env, self.FLAGS = self.wtm_env_levy_style(kwargs['make_env'], self.FLAGS)
        self.check_envs(env, wtm_env)

        self.agent = wtm_agent
        self.env = wtm_env

    def check_envs(self, env, wtm_env):
        #  assert env.model == wtm_env.model
        assert type(env.sim) == type(wtm_env.sim)
        assert env.state_dim == wtm_env.state_dim
        assert env.action_dim == wtm_env.action_dim
        assert (env.action_bounds == wtm_env.action_bounds).all()
        assert (env.action_offset == wtm_env.action_offset).all()
        assert env.end_goal_dim == wtm_env.end_goal_dim
        assert env.subgoal_dim == wtm_env.subgoal_dim
        assert (env.subgoal_bounds == wtm_env.subgoal_bounds).all()
        assert (env.subgoal_bounds_symmetric == wtm_env.subgoal_bounds_symmetric).all()
        assert (env.subgoal_bounds_offset == wtm_env.subgoal_bounds_offset).all()
        assert env.max_actions == wtm_env.max_actions
        print('PASSED ASSERTS')

    def wtm_env_levy_style(self,make_env, FLAGS):
        env = make_env().env
        print(type(env), dir(env))

        # design_agent_and_env
        FLAGS.layers = 2
        if FLAGS.time_scale == 0:
            # Enter max sequence length in which each policy will specialize
            FLAGS.time_scale = 30

        max_actions = 700
        max_actions = FLAGS.time_scale**(FLAGS.layers)
        timesteps_per_action = 15
        env.max_actions = max_actions
        env.visualize = False

        env.action_bounds = env.sim.model.actuator_ctrlrange[:,1]
        env.action_offset = np.zeros((len(env.action_bounds)))
        env.action_dim = len(env.sim.model.actuator_ctrlrange)
        env.end_goal_dim = len(env.goal_space_test)
        env.subgoal_dim = len(env.subgoal_bounds)
        env.state_dim = self.input_dims['o']
        env.subgoal_bounds_symmetric = np.zeros((len(env.subgoal_bounds)))
        env.subgoal_bounds_offset = np.zeros((len(env.subgoal_bounds)))

        for i in range(len(env.subgoal_bounds)):
            env.subgoal_bounds_symmetric[i] = (env.subgoal_bounds[i][1] - env.subgoal_bounds[i][0])/2
            env.subgoal_bounds_offset[i] = env.subgoal_bounds[i][1] - env.subgoal_bounds_symmetric[i]

        def next_goal(test):
            end_goal = np.zeros((len(env.goal_space_test)))
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

            return end_goal


        env.get_next_goal = next_goal

        def reset_sim(next_goal = None):

            # Reset controls
            env.sim.data.ctrl[:] = 0

            if env.name == "ant_reacher.xml":
                while True:
                    # Reset joint positions and velocities
                    for i in range(len(env.sim.data.qpos)):
                        env.sim.data.qpos[i] = np.random.uniform(env.initial_state_space[i][0],env.initial_state_space[i][1])

                    for i in range(len(env.sim.data.qvel)):
                        env.sim.data.qvel[i] = np.random.uniform(
                            env.initial_state_space[len(env.sim.data.qpos) + i][0],
                            env.initial_state_space[len(env.sim.data.qpos) + i][1])

                    # Ensure initial ant position is more than min_dist away from goal
                    min_dist = 8
                    if np.linalg.norm(next_goal[:2] - env.sim.data.qpos[:2]) > min_dist:
                        break

            elif env.name == "ant_four_rooms.xml":

                # Choose initial start state to be different than room containing the end goal

                # Determine which of four rooms contains goal
                goal_room = 0

                if next_goal[0] < 0 and next_goal[1] > 0:
                    goal_room = 1
                elif next_goal[0] < 0 and next_goal[1] < 0:
                    goal_room = 2
                elif next_goal[0] > 0 and next_goal[1] < 0:
                    goal_room = 3


                # Place ant in room different than room containing goal
                # initial_room = (goal_room + 2) % 4


                initial_room = np.random.randint(0,4)
                while initial_room == goal_room:
                    initial_room = np.random.randint(0,4)


                # Set initial joint positions and velocities
                for i in range(len(env.sim.data.qpos)):
                    env.sim.data.qpos[i] = np.random.uniform(env.initial_state_space[i][0],env.initial_state_space[i][1])

                for i in range(len(env.sim.data.qvel)):
                    env.sim.data.qvel[i] = np.random.uniform(env.initial_state_space[len(env.sim.data.qpos) + i][0],env.initial_state_space[len(env.sim.data.qpos) + i][1])

                # Move ant to correct room
                env.sim.data.qpos[0] = np.random.uniform(3,6.5)
                env.sim.data.qpos[1] = np.random.uniform(3,6.5)

                # If goal should be in top left quadrant
                if initial_room == 1:
                    env.sim.data.qpos[0] *= -1

                # Else if goal should be in bottom left quadrant
                elif initial_room == 2:
                    env.sim.data.qpos[0] *= -1
                    env.sim.data.qpos[1] *= -1

                # Else if goal should be in bottom right quadrant
                elif initial_room == 3:
                    env.sim.data.qpos[1] *= -1

                # print("Goal Room: %d" % goal_room)
                # print("Initial Ant Room: %d" % initial_room)

            else:

                # Reset joint positions and velocities
                for i in range(len(env.sim.data.qpos)):
                    env.sim.data.qpos[i] = np.random.uniform(env.initial_state_space[i][0],env.initial_state_space[i][1])

                for i in range(len(env.sim.data.qvel)):
                    env.sim.data.qvel[i] = np.random.uniform(
                        env.initial_state_space[len(env.sim.data.qpos) + i][0],
                        env.initial_state_space[len(env.sim.data.qpos) + i][1])

            env.sim.step()

            # Return state
            return np.concatenate((env.sim.data.qpos, env.sim.data.qvel))

        env.reset_sim = reset_sim

        def exe_action(action):
            env.sim.data.ctrl[:] = action
            env.sim.step()

            if env.visualize:
                env.render()

            return np.concatenate((env.sim.data.qpos, env.sim.data.qvel))


        env.execute_action = exe_action

        env.project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))
        env.velo_threshold = 0.8
        env.subgoal_thresholds = np.concatenate((env.end_goal_thresholds ,[env.velo_threshold, env.velo_threshold]))

        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["subgoal_penalty"] = -FLAGS.time_scale
        agent_params["atomic_noise"] = [0.1 for i in range(8)]
        agent_params["subgoal_noise"] = [0.1 for i in range(len(env.sub_goal_thresholds))]
        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100
        FLAGS.id = 1
        agent = Agent(FLAGS,env,agent_params)

        return agent, env, FLAGS


    def init_levy(self, FLAGS):
        env_import_name = "baselines.hac.env_designs.ANT_FOUR_ROOMS_2_design_agent_and_env"
        design_agent_and_env_module = importlib.import_module(env_import_name)
        # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
        FLAGS.id = 0
        agent, env = design_agent_and_env_module.design_agent_and_env(FLAGS)
        return agent, env




