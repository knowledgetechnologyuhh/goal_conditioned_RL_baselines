import numpy as np
import random

from gym.envs.robotics import rotations
from wtm_envs.mujoco import robot_env, utils
from mujoco_py.generated import const as mj_const
from wtm_envs.mujoco.hook_env_pddl import *
from wtm_envs.mujoco.wtm_env import goal_distance
from wtm_envs.mujoco.wtm_env import WTMEnv
from wtm_envs.mujoco.hook_env_pddl import PDDLHookEnv
import mujoco_py


class AntEnv(WTMEnv):
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

        #num_actions = len(self.sim.model.actuator_ctrlrange)



        WTMEnv.__init__(self, model_path=model_path, n_substeps=n_substeps, initial_qpos=self.initial_state_space,
                        is_fetch_env=False)
        # PDDLHookEnv.__init__(self, n_objects=self.n_objects)

        self.distance_threshold = self.end_goal_thresholds
        # TODO: use separate distance threshold for subgoals (it already exists, but is not used yet)

        #self.reset()

    # Execute low-level action for number of frames specified by num_frames_skip
    # def execute_action(self, action):
    #    self.sim.data.ctrl[:] = action
    #    for _ in range(self.num_frames_skip):
    #        self.sim.step()
    #        if self.visualize:
    #            self.viewer.render()
    #    return self.get_state()

    def _set_action(self, action):
        # Apply action to simulation.
        self.sim.data.ctrl[:] = action # from the Levy code
        #utils.ctrl_set_action(self.sim, action)
        #utils.mocap_set_action(self.sim, action)
        self.step_ctr += 1

    def _obs2goal(self, obs):
        return self.project_state_to_end_goal(self.sim, obs)

    def _obs2subgoal(self, obs):
        return self.project_state_to_sub_goal(self.sim, obs)
        # TODO: Not used.


    # Get state, which concatenates joint positions and velocities
    def _get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    def _get_obs(self, grip_pos=None, grip_velp=None):
        # If the grip position and grip velp are provided externally, the external values will be used.
        # This can later be extended to provide the properties of all elements in the scene.
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        obs = self._get_state()

        noisy_obs = self.add_noise(obs.copy(), self.obs_history, self.obs_noise_coefficient)
        achieved_goal = self._obs2goal(noisy_obs)

        obs = {'observation': noisy_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy(),
               'non_noisy_obs': obs.copy()}
        # obs['achieved_goal'] = self._obs2goal(obs['observation'])

        return obs

    def _viewer_setup(self, mode='human'):
        return
    # TODO: if learning from pixels should be enabled, the camera needs to look from the bird's eye view

    def compute_reward(self, achieved_goal, goal, info):
        individual_differences = achieved_goal - goal
        d = np.linalg.norm(individual_differences, axis=-1)

        if self.reward_type == 'sparse':
            reward = -1 * np.any(np.abs(individual_differences) > self.distance_threshold, axis=-1).astype(np.float32)

            #print("Actual goal: ", goal, " achieved: ", achieved_goal, " reward: ", reward)
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

        for i in range(1, min(len(subgoals) + 1, 11)):
            self.sim.data.mocap_pos[i][:3] = np.copy(subgoals[subgoal_ind][:3])
            self.sim.model.site_rgba[i][3] = 1

            subgoal_ind += 1

    def _render_callback(self):
        #print(self.final_goal)
        #print(self.goal)

        if self.final_goal != []:
            self.display_end_goal(self.final_goal)

        self.display_subgoals([self.goal])
        # Visualize target.
        # sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        #
        # obj_goal_start_idx = 0
        # if self.gripper_goal != 'gripper_none':
        #     gripper_target_site_id = self.sim.model.site_name2id('final_arm_target')
        #     gripper_goal_site_id = self.sim.model.site_name2id('final_arm_goal')
        #     gripper_tgt_size = (np.ones(3) * 0.02)
        #     gripper_tgt_size[1] = 0.05
        #     self.sim.model.site_size[gripper_target_site_id] = gripper_tgt_size
        #     self.sim.model.site_size[gripper_goal_site_id] = gripper_tgt_size
        #     if self.goal != []:
        #         gripper_tgt_goal = self.goal[0:3] - sites_offset[0]
        #         self.sim.model.site_pos[gripper_target_site_id] = gripper_tgt_goal
        #     if self.final_goal != []:
        #         gripper_tgt_final_goal = self.final_goal[0:3] - sites_offset[0]
        #         self.sim.model.site_pos[gripper_goal_site_id] = gripper_tgt_final_goal
        #     obj_goal_start_idx += 3
        #
        # for n in range(self.n_objects):
        #     if n == 0:
        #         o_tgt_y = 0.08
        #     else:
        #         o_tgt_y = 0.02
        #     o_target_site_id = self.sim.model.site_name2id('target{}'.format(n))
        #     o_goal_site_id = self.sim.model.site_name2id('goal{}'.format(n))
        #     o_tgt_size = (np.ones(3) * 0.02)
        #     o_tgt_size[1] = o_tgt_y
        #     self.sim.model.site_size[o_target_site_id] = o_tgt_size
        #     self.sim.model.site_size[o_goal_site_id] = o_tgt_size
        #     if self.goal != []:
        #         o_tgt_goal = self.goal[obj_goal_start_idx:obj_goal_start_idx + 3] - sites_offset[0]
        #         self.sim.model.site_pos[o_target_site_id] = o_tgt_goal
        #     if self.final_goal != []:
        #         o_tgt_final_goal = self.final_goal[obj_goal_start_idx:obj_goal_start_idx + 3] - sites_offset[0]
        #         self.sim.model.site_pos[o_goal_site_id] = o_tgt_final_goal
        #
        #     obj_goal_start_idx += 3
        #
        # self.sim.forward()

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

        # Visualize End Goal
        #self.display_end_goal(end_goal)

        return end_goal

    def _env_setup(self, initial_qpos):
        pass
        #
        # for name, value in initial_qpos.items():
        #     self.sim.data.set_joint_qpos(name, value)
        # utils.reset_mocap_welds(self.sim)
        # self.sim.forward()
        #
        # # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) \
        #                  + self.sim.data.get_site_xpos('robot0:grip')
        # gripper_rotation = np.array([1., 0., 1., 0.])
        # self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        # self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # for _ in range(10):
        #     self.sim.step()
        #
        # # offset the random goal if gripper random is used
        # # self.random_gripper_goal_pos_offset = (0.2, 0.0, 0.0)
        # self.random_gripper_goal_pos_offset = (0.0, 0.0, 0.14)
        #
        # # Extract information for sampling goals.
        # self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        # if self.n_objects > 0:
        #     self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def get_scale_and_offset_for_normalized_subgoal(self):
        return self.goal_space_scale, self.goal_space_offset

        # n_objects = self.n_objects
        # obj_height = self.obj_height
        # scale_xy = self.target_range
        # scale_z = obj_height * n_objects / 2
        # scale = np.array([scale_xy, scale_xy, scale_z] * (n_objects + 1))
        # offset = np.array(list(self.initial_gripper_xpos) * (n_objects + 1))
        # for j, off in enumerate(offset):
        #     if j == 2:
        #         offset[j] += self.random_gripper_goal_pos_offset[2]
        #         if self.gripper_goal == 'gripper_random':
        #             scale[j] = self.target_range
        #     elif (j + 1) % 3 == 0:
        #         offset[j] += obj_height * n_objects / 2
        # if self.gripper_goal == 'gripper_none':
        #     scale = scale[3:]
        #     offset = offset[3:]
        # return scale, offset
