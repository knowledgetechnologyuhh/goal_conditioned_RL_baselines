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

    def __init__(
            self, model_path, n_substeps, reward_type, name):
        """Initializes a new Ant environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.name = ""
        self.step_ctr = 0

        self.obs_limits = [None, None]
        self.obs_noise_coefficient = 0.0

        self.plan_cache = {}
        self.goal_hierarchy = {}
        self.goal = []
        self.goal_size = 3
        self.final_goal = []

        self._viewers = {}

        # Provide initial state space consisting of the ranges for all joint angles and velocities.
        # In the Ant Reacher task, we use a random initial torso position and use fixed values for the remainder.
        self.initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        self.initial_joint_pos = np.reshape(self.initial_joint_pos, (len(self.initial_joint_pos), 1))
        self.initial_joint_ranges = np.concatenate((self.initial_joint_pos, self.initial_joint_pos), 1)
        self.initial_joint_ranges[0] = np.array([-6, 6])
        self.initial_joint_ranges[1] = np.array([-6, 6])

        # Concatenate velocity ranges
        self.initial_state_space = np.concatenate(
            (self.initial_joint_ranges, np.zeros((len(self.initial_joint_ranges) - 1, 2))), 0)

        # Provide end goal space.
        max_range = 6
        self.goal_space_train = [[-max_range, max_range], [-max_range, max_range], [0.45, 0.55]]
        self.goal_space_test = [[-max_range, max_range], [-max_range, max_range], [0.45, 0.55]]

        # Provide a function that maps from the state space to the end goal space.
        # This is used to
        # (i)  determine whether the agent should be given the sparse reward and
        # (ii) for Hindsight Experience Replay to determine which end goal was achieved after a sequence of actions.
        self.project_state_to_end_goal = lambda sim, state: state[:3]

        # Set end goal achievement thresholds.  If the agent is within the threshold for each dimension,
        # the end goal has been achieved and the reward of 0 is granted.
        # For the Ant Reacher task, the end goal will be the desired (x,y) position of the torso
        len_threshold = 0.4
        height_threshold = 0.2
        self.end_goal_thresholds = np.array([len_threshold, len_threshold, height_threshold])

        # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.
        # Subgoal space can be the same as the state space or some other projection out of the state space.
        # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
        cage_max_dim = 8
        max_height = 1
        max_velo = 3
        subgoal_bounds = np.array(
            [[-cage_max_dim, cage_max_dim], [-cage_max_dim, cage_max_dim], [0, max_height], [-max_velo, max_velo],
             [-max_velo, max_velo]])

        # Provide state to subgoal projection function.
        # a = np.concatenate((sim.data.qpos[:2], np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in range(3)])))
        project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array(
            [1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array(
            [3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))

        # Set subgoal achievement thresholds
        velo_threshold = 0.8
        quat_threshold = 0.5
        # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
        subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])

        num_actions = 8

        WTMEnv.__init__(self, model_path=model_path, n_substeps=n_substeps, initial_qpos=self.initial_state_space,
                        num_actions=num_actions, is_fetch_env=False)
        # PDDLHookEnv.__init__(self, n_objects=self.n_objects)

        self.distance_threshold = self.end_goal_thresholds
        # TODO: use separate distance threshold for subgoals (it already exists, but is not used yet)

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
        #self.sim.data.ctrl[:] = action # from the Levy code
        utils.ctrl_set_action(self.sim, action)
        #utils.mocap_set_action(self.sim, action)
        self.step_ctr += 1

    def _obs2goal(self, obs):
        return self.project_state_to_end_goal(self.sim, obs)

    # TODO: what about projections to sub goals?

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
        self.display_end_goal(self.goal)
        self.display_subgoals(self.final_goal)
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

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal=None):
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
        return self._get_state()

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
        self.display_end_goal(end_goal)

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
        return 1, np.zeros_like(self.goal)
        # TODO: this obviously does not normalize, check out how it can be done for this env.

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
