import numpy as np
import random

from gym.envs.robotics import rotations
from wtm_envs.mujoco import robot_env, utils
from mujoco_py.generated import const as mj_const
from wtm_envs.mujoco.hook_env_pddl import *
from wtm_envs.mujoco.wtm_env import goal_distance
from wtm_envs.mujoco.wtm_env import WTMEnv
from wtm_envs.mujoco.ant_env_pddl import PDDLAntEnv
import mujoco_py


class AntEnv(WTMEnv, PDDLAntEnv):
    """Superclass for all Ant environments.
    """

    #goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds,


    def __init__(
            self, model_path, n_substeps, reward_type, name, goal_space_train, goal_space_test,
            project_state_to_end_goal, project_state_to_subgoal, end_goal_thresholds, initial_state_space,
            subgoal_bounds, subgoal_thresholds):
        """Initializes a new Ant environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.name = name
        self.step_ctr = 0
        self.reward_type = reward_type

        self.obs_limits = [None, None]
        self.obs_noise_coefficient = 0.0

        self.plan_cache = {}
        self.goal_hierarchy = {}
        self.goal = []
        self.goal_size = 3
        self.final_goal = []

        self._viewers = {}

        self.initial_state_space = initial_state_space
        self.end_goal_thresholds = end_goal_thresholds
        self.sub_goal_thresholds = subgoal_thresholds
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_sub_goal = project_state_to_subgoal
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_bounds = subgoal_bounds

        self.goal_space_offset = [np.mean(limits) for limits in goal_space_train]
        self.goal_space_scale = [goal_space_train[limits_idx][1] - self.goal_space_offset[limits_idx]
                            for limits_idx in range(len(goal_space_train))]

        n_actions = 8 #len(self.sim.model.actuator_ctrlrange)

        WTMEnv.__init__(self, model_path=model_path, n_substeps=n_substeps, initial_qpos=self.initial_state_space,
                        n_actions=n_actions)
        PDDLAntEnv.__init__(self)

        # self.distance_threshold = self.end_goal_thresholds
        # TODO: use separate distance threshold for subgoals (it already exists, but is not used yet)

    def _set_action(self, action):
        # Apply action to simulation.
        self.sim.data.ctrl[:] = action
        self.step_ctr += 1

    def _obs2goal(self, obs):
        return self.project_state_to_end_goal(self.sim, obs)

    def _obs2subgoal(self, obs):
        return self.project_state_to_sub_goal(self.sim, obs)
        # TODO: Not used, because atm sub goal mapping is equivalent to goal mapping

    # Get state, which concatenates joint positions and velocities
    def _get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def _get_obs(self, grip_pos=None, grip_velp=None):
        obs = self._get_state()
        ## Torso qpos refers to the first three coordinates of qpos.
        # torso_qpos = self.sim.data.get_joint_qpos('root')

        # torso_id = self.sim.model.body_name2id('torso')
        # torso_pos = self.sim.data.get_body_xpos('torso')
        # torso_quat = self.sim.data.get_body_xquat('torso')
        # torso_geom_pos = self.sim.data.get_geom_xpos('torso_geom')

        noisy_obs = self.add_noise(obs.copy(), self.obs_history, self.obs_noise_coefficient)
        achieved_goal = self._obs2goal(noisy_obs)

        obs = {'observation': noisy_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy(),
               'non_noisy_obs': obs.copy()}
        return obs

    def _viewer_setup(self, mode='human'):
        return
        # TODO: if learning from pixels should be enabled, the camera needs to look from the bird's eye view

    def compute_reward(self, achieved_goal, goal, info):
        individual_differences = achieved_goal - goal
        d = np.linalg.norm(individual_differences, axis=-1)

        if self.reward_type == 'sparse':
            reward = -1 * np.any(np.abs(individual_differences) > self.distance_threshold, axis=-1).astype(np.float32)
            return reward
        else:
            return -1 * d

    def _is_success(self, achieved_goal, desired_goal):
        d = np.abs(achieved_goal - desired_goal)
        return np.all(d < self.distance_threshold, axis=-1).astype(np.float32)

    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self, end_goal):
        # Goal can be visualized by changing the location of the relevant site object.
        self.sim.data.mocap_pos[0][:3] = np.copy(end_goal[:3])

    # Visualize all subgoals
    def display_subgoals(self, subgoals):
        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            self.sim.data.mocap_pos[i][:3] = np.copy(subgoals[subgoal_ind][:3])
            self.sim.model.site_rgba[i][3] = 1

            subgoal_ind += 1

    def _render_callback(self):
        if self.final_goal != []:
            self.display_end_goal(self.final_goal)

        self.display_subgoals([self.goal])
        #TODO: as soon as multiple subgoals should be visualized, they should be in a list in self.goal.
        #TODO  Then, the list cast above around self.goal needs to be removed

    def reset(self):
        self.goal = self._sample_goal().copy()
        obs = self._reset_sim(self.goal)
        return obs


    # Reset simulation to state within initial state specified by user
    def _reset_sim(self, next_goal=None):
        self.step_ctr = 0

        # Reset controls
        self.sim.data.ctrl[:] = 0
        if self.name == "ant_reacher.xml":
            while True:
                # Reset joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],
                                                              self.initial_state_space[i][1])

                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],
                                                              self.initial_state_space[len(self.sim.data.qpos) + i][1])

                # Ensure initial ant position is more than min_dist away from goal
                min_dist = 8
                if np.linalg.norm(next_goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                    break
        elif self.name == "ant_four_rooms.xml":
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

            initial_room = np.random.randint(0, 4)
            while initial_room == goal_room:
                initial_room = np.random.randint(0, 4)

            # Set initial joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],
                                                          self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],
                                                          self.initial_state_space[len(self.sim.data.qpos) + i][1])

            # Move ant to correct room
            self.sim.data.qpos[0] = np.random.uniform(3, 6.5)
            self.sim.data.qpos[1] = np.random.uniform(3, 6.5)

            # If goal should be in top left quadrant
            if initial_room == 1:
                self.sim.data.qpos[0] *= -1

            # Else if goal should be in bottom left quadrant
            elif initial_room == 2:
                self.sim.data.qpos[0] *= -1
                self.sim.data.qpos[1] *= -1

            # Else if goal should be in bottom right quadrant
            elif initial_room == 3:
                self.sim.data.qpos[1] *= -1

            # print("Goal Room: %d" % goal_room)
            # print("Initial Ant Room: %d" % initial_room)
        self.sim.step()

        # Return state
        return self._get_obs()

    # Function returns an end goal
    def _sample_goal(self):
        end_goal = np.zeros((len(self.goal_space_test)))

        # Randomly select one of the four rooms in which the goal will be located
        room_num = np.random.randint(0, 4)

        # Pick exact goal location
        end_goal[0] = np.random.uniform(3, 6.5)
        end_goal[1] = np.random.uniform(3, 6.5)
        end_goal[2] = np.random.uniform(0.45, 0.55)

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

    def _env_setup(self, initial_qpos):
        # not necessary for this env
        pass

    def get_scale_and_offset_for_normalized_subgoal(self):
        return self.goal_space_scale, self.goal_space_offset
