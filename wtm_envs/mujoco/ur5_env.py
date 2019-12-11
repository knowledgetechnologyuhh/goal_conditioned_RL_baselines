import numpy as np
import random

from gym.envs.robotics import rotations
from wtm_envs.mujoco import robot_env, utils
from mujoco_py.generated import const as mj_const

from wtm_envs.mujoco.wtm_env import WTMEnv
import mujoco_py


class UR5Env(WTMEnv):
    """Superclass for all UR5 environment
    """

    def __init__(
        self, model_path, n_substeps, reward_type, name, goal_space_train, goal_space_test,
            project_state_to_end_goal, project_state_to_subgoal, end_goal_thresholds, initial_state_space,
            initial_joint_pos,
            subgoal_bounds, subgoal_thresholds, obs_type=1
    ):
        """
        UR5 environment
        :param model_path:
        :param n_substeps:
        :param reward_type:
        :param name:
        :param goal_space_train:
        :param goal_space_test:
        :param project_state_to_end_goal:
        :param project_state_to_subgoal:
        :param end_goal_thresholds:
        :param initial_state_space:
        :param subgoal_bounds:
        :param subgoal_thresholds:
        """

        # assert n_objects == 2, "Cannot have more than 2 objects for this environment at the time being!"

        # self.gripper_extra_height = gripper_extra_height
        # self.block_gripper = block_gripper
        # self.target_in_the_air = target_in_the_air
        # self.target_offset = target_offset
        self.reward_type = reward_type

        # self.gripper_goal = gripper_goal

        self.name = name
        self.step_ctr = 0
        self.reward_type = reward_type

        self.obs_limits = [None, None]
        self.obs_noise_coefficient = 0.0

        self.goal_hierarchy = {}
        self.goal = []
        self.end_goal_dim = self.goal_size = len(goal_space_test)
        self.final_goal = []

        self._viewers = {}

        self.initial_joint_pos = initial_joint_pos
        self.initial_state_space = initial_state_space
        self.end_goal_thresholds = end_goal_thresholds
        self.sub_goal_thresholds = subgoal_thresholds
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_sub_goal = project_state_to_subgoal
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_bounds = subgoal_bounds

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # These are similar to original Levy implementation
        self.goal_space_offset = self.subgoal_bounds_offset
        self.goal_space_scale = self.subgoal_bounds_symmetric

        # # TODO need to check these
        # self.goal_space_offset = [np.mean(limits) for limits in goal_space_train]
        # self.goal_space_scale = [goal_space_train[limits_idx][1] - self.goal_space_offset[limits_idx]
        #                          for limits_idx in range(len(goal_space_train))]

        if obs_type == 1:
            self.visual_input = False
        else:
            self.visual_input = True

        WTMEnv.__init__(self, model_path=model_path, n_substeps=n_substeps, initial_qpos=self.initial_state_space,
                        n_actions=3)

        self.action_bounds = self.sim.model.actuator_ctrlrange[:, 1]  # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds)))  # Assumes symmetric low-level action ranges

        self.action_space.low = -self.action_bounds
        self.action_space.high = self.action_bounds

        if obs_type == 2:
            self.camera_name = 'external_camera_1'
        elif obs_type == 3:
            self.camera_name = 'internal_camera_r'

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(10):
            self._set_action(action)
            self.sim.step()
            self._step_callback()
        obs = self._get_obs()

        done = False
        is_success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info = {
            'is_success': is_success
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        return obs, reward, done, info

    def _set_action(self, action):
        # assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # action = action*self.action_bounds + self.action_offset

        self.sim.data.ctrl[:] = action
        # pos_ctrl, gripper_ctrl = action[:3], action[3]
        #
        # pos_ctrl *= 0.05  # limit maximum change in position
        # rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # assert gripper_ctrl.shape == (2,)
        # if self.block_gripper:
        #     gripper_ctrl = np.zeros_like(gripper_ctrl)
        # action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        #
        # # Apply action to simulation.
        # utils.ctrl_set_action(self.sim, action)
        # utils.mocap_set_action(self.sim, action)
        self.step_ctr += 1

    def _obs2goal(self, obs):
        return self.project_state_to_end_goal(self.sim, obs)

    def _get_obs(self, grip_pos=None, grip_velp=None):
        # If the grip position and grip velp are provided externally, the external values will be used.
        # This can later be extended to provide the properties of all elements in the scene.
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # # positions
        # if grip_pos is None:
        #     grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        # if grip_velp is None:
        #     grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        # robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        # obs = np.concatenate([robot_qpos, robot_qvel])

        if self.visual_input:
            # image_obs = self._get_image()
            image_obs = self.offscreen_buffer()
            image_obs = image_obs.reshape(-1)
            obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel, image_obs])
            noisy_obs = obs.copy()
        else:
            obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel])
            noisy_obs = self.add_noise(obs.copy(), self.obs_history, self.obs_noise_coefficient)
        achieved_goal = self._obs2goal(noisy_obs)

        obs = {'observation': noisy_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy(), 'non_noisy_obs': obs.copy()}
        # obs['achieved_goal'] = self._obs2goal(obs['observation'])

        return obs

    # def _viewer_setup(self,mode='human'):
    #     if mode == 'human':
    #         body_id = self.sim.model.body_name2id('robot0:gripper_link')
    #         lookat = self.sim.data.body_xpos[body_id]
    #         for idx, value in enumerate(lookat):
    #             self._viewers[mode].cam.lookat[idx] = value
    #         self._viewers[mode].cam.distance = 2.5
    #         self._viewers[mode].cam.azimuth = 132.
    #         self._viewers[mode].cam.elevation = -14.
    #     elif mode == 'rgb_array':
    #         body_id = self.sim.model.body_name2id('robot0:gripper_link')
    #         lookat = self.sim.data.body_xpos[body_id]
    #         for idx, value in enumerate(lookat):
    #             self._viewers[mode].cam.lookat[idx] = value
    #         self._viewers[mode].cam.distance = 1.
    #         self._viewers[mode].cam.azimuth = 180.
    #         self._viewers[mode].cam.elevation = -40.

    def _render_callback(self):
        # Visualize target.
        if self.final_goal != []:
            self.display_end_goal(self.final_goal)
        # self.display_end_goal(self.goal)

        # TODO check this
        self.sim.forward()

    def _reset_sim(self):
        self.step_ctr = 0

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # self.sim.set_state(self.initial_state)

        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = self.initial_joint_pos[i] \
                                    + np.random.uniform(self.initial_state_space[i][0], self.initial_state_space[i][1])
        #
        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.forward()
        self.sim.step()
        return True

    def _sample_goal(self):
        # obs = self._get_obs()
        end_goal = np.zeros((len(self.goal_space_test)))

        if self.name == "ur5.xml":

            goal_possible = False
            while not goal_possible:
                end_goal = np.zeros(shape=(self.end_goal_dim,))
                end_goal[0] = np.random.uniform(self.goal_space_test[0][0],self.goal_space_test[0][1])

                end_goal[1] = np.random.uniform(self.goal_space_test[1][0],self.goal_space_test[1][1])
                end_goal[2] = np.random.uniform(self.goal_space_test[2][0],self.goal_space_test[2][1])

                # Next need to ensure chosen joint angles result in achievable task (i.e., desired end effector position
                # is above ground)

                theta_1 = end_goal[0]
                theta_2 = end_goal[1]
                theta_3 = end_goal[2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0,0.13585,0,1])
                forearm_pos_3 = np.array([0.425,0,0,1])
                wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                # Make sure wrist 1 pos is above ground so can actually be reached
                if np.absolute(end_goal[0]) > np.pi/4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                    goal_possible = True

        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])

        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal.copy()


    # def _is_success(self, achieved_goal, desired_goal):
    #     d = goal_distance(achieved_goal, desired_goal)
    #     return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal, info):    # TODO check
        individual_differences = achieved_goal - goal
        d = np.linalg.norm(individual_differences, axis=-1)

        if self.reward_type == 'sparse':
            reward = -1 * np.any(np.abs(individual_differences) > self.end_goal_thresholds, axis=-1).astype(np.float32)
            return reward
        else:
            return -1 * d

    def _is_success(self, achieved_goal, desired_goal): # TODO check
        d = np.abs(achieved_goal - desired_goal)
        return np.all(d < self.end_goal_thresholds, axis=-1).astype(np.float32)

    def _env_setup(self, initial_qpos):
        pass

    def _viewer_setup(self, mode='human'):
        if mode == 'human':
            body_id = self.sim.model.body_name2id('upper_arm_link')
            lookat = self.sim.data.body_xpos[body_id]
            for idx, value in enumerate(lookat):
                self._viewers[mode].cam.lookat[idx] = value
            self._viewers[mode].cam.distance = 2.5
            self._viewers[mode].cam.azimuth = 132.
            self._viewers[mode].cam.elevation = -14.
        elif mode == 'rgb_array':
            body_id = self.sim.model.body_name2id('upper_arm_link')
            lookat = self.sim.data.body_xpos[body_id]
            for idx, value in enumerate(lookat):
                self._viewers[mode].cam.lookat[idx] = value
            self._viewers[mode].cam.distance = 1.7
            self._viewers[mode].cam.azimuth = 180.
            self._viewers[mode].cam.elevation = -50.

    def get_scale_and_offset_for_normalized_subgoal(self):
        return self.goal_space_scale, self.goal_space_offset

    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self, end_goal):

        # Goal can be visualized by changing the location of the relevant site object.
        if self.name == "pendulum.xml":
            self.sim.data.mocap_pos[0] = np.array([0.5 * np.sin(end_goal[0]), 0, 0.5 * np.cos(end_goal[0]) + 0.6])
        elif self.name == "ur5.xml":

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array(
                [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
                 [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0], [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0], [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array(
                [[np.cos(theta_3), 0, np.sin(theta_3), 0.425], [0, 1, 0, 0], [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                 [0, 0, 0, 1]])

            # Determine joint position relative to original reference frame
            # shoulder_pos = T_1_0.dot(shoulder_pos_1)
            upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

            """
            print("\nEnd Goal Joint Pos: ")
            print("Upper Arm Pos: ", joint_pos[0])
            print("Forearm Pos: ", joint_pos[1])
            print("Wrist Pos: ", joint_pos[2])
            """

            for i in range(3):
                self.sim.data.mocap_pos[i] = joint_pos[i]

        else:
            assert False, "Provide display end goal function in environment.py file"

    # Visualize all subgoals
    def display_subgoals(self, subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1, min(len(subgoals), 11)):
            if self.name == "pendulum.xml":
                self.sim.data.mocap_pos[i] = np.array(
                    [0.5 * np.sin(subgoals[subgoal_ind][0]), 0, 0.5 * np.cos(subgoals[subgoal_ind][0]) + 0.6])
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1

            elif self.name == "ur5.xml":

                theta_1 = subgoals[subgoal_ind][0]
                theta_2 = subgoals[subgoal_ind][1]
                theta_3 = subgoals[subgoal_ind][2]

                # shoulder_pos_1 = np.array([0,0,0,1])
                upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
                forearm_pos_3 = np.array([0.425, 0, 0, 1])
                wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

                # Transformation matrix from shoulder to base reference frame
                T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

                # Transformation matrix from upper arm to shoulder reference frame
                T_2_1 = np.array(
                    [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
                     [0, 0, 0, 1]])

                # Transformation matrix from forearm to upper arm reference frame
                T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0], [0, 1, 0, 0.13585],
                                  [-np.sin(theta_2), 0, np.cos(theta_2), 0], [0, 0, 0, 1]])

                # Transformation matrix from wrist 1 to forearm reference frame
                T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425], [0, 1, 0, 0],
                                  [-np.sin(theta_3), 0, np.cos(theta_3), 0], [0, 0, 0, 1]])

                # Determine joint position relative to original reference frame
                # shoulder_pos = T_1_0.dot(shoulder_pos_1)
                upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
                forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
                wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

                joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

                """
                print("\nSubgoal %d Joint Pos: " % i)
                print("Upper Arm Pos: ", joint_pos[0])
                print("Forearm Pos: ", joint_pos[1])
                print("Wrist Pos: ", joint_pos[2])
                """

                # Designate site position for upper arm, forearm and wrist
                for j in range(3):
                    self.sim.data.mocap_pos[3 + 3 * (i - 1) + j] = np.copy(joint_pos[j])
                    self.sim.model.site_rgba[3 + 3 * (i - 1) + j][3] = 1

                # print("\nLayer %d Predicted Pos: " % i, wrist_1_pos[:3])

                subgoal_ind += 1
            else:
                # Visualize desired gripper position, which is elements 18-21 in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1


