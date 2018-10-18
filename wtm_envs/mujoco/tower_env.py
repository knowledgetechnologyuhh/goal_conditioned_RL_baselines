import numpy as np
import random

from gym.envs.robotics import rotations
from wtm_envs.mujoco import robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class TowerEnv(robot_env.RobotEnv):
    """Superclass for all Tower environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height=None, max_tower_height=None,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            gripper_goal ('gripper_none', 'gripper_above', 'gripper_random'): the gripper's goal location
            n_objects (int): no of objects in the environment. If none, then no_of_objects=0
            min_tower_height (int): the minimum height of the tower.
            max_tower_height (int): the maximum height of the tower.
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        self.gripper_goal = gripper_goal
        self.n_objects = n_objects
        self.table_height = table_height
        self.obj_height = obj_height
        self.min_tower_height = min_tower_height
        self.max_tower_height = max_tower_height

        self.goal = []
        self.goal_size = (n_objects * 3)
        if self.gripper_goal != 'gripper_none':
            self.goal_size += 3
        self.gripper_has_target = (gripper_goal != 'gripper_none')

        super(TowerEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos, object_rot, object_velp, object_velr = ([] for _ in range(4))
        object_rel_pos = []

        if self.n_objects > 0:
            for n_o in range(self.n_objects):
                oname = 'object{}'.format(n_o)
                this_object_pos = self.sim.data.get_site_xpos(oname)
                # rotations
                this_object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(oname))
                # velocities
                this_object_velp = self.sim.data.get_site_xvelp(oname) * dt
                this_object_velr = self.sim.data.get_site_xvelr(oname) * dt
                # gripper state
                this_object_rel_pos = this_object_pos - grip_pos
                this_object_velp -= grip_velp

                object_pos = np.concatenate([object_pos, this_object_pos])
                object_rot = np.concatenate([object_rot, this_object_rot])
                object_velp = np.concatenate([object_velp, this_object_velp])
                object_velr = np.concatenate([object_velr, this_object_velr])
                object_rel_pos = np.concatenate([object_rel_pos, this_object_rel_pos])
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.array(np.zeros(3))

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        gripper_achieved_goal = obs[:3]
        target_achieved_goal = obs[3:6]

        if not self.gripper_has_target:
            gripper_achieved_goal = []

        achieved_goal = np.concatenate([gripper_achieved_goal, target_achieved_goal])

        obs = {'observation': obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy()}

        if self.gripper_goal != 'gripper_none':
            obs['achieved_goal'] = obs['observation'][:self.goal_size]
        else:
            obs['achieved_goal'] = obs['observation'][3:self.goal_size+3]

        return obs

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        obj_goal_start_idx = 0
        if self.gripper_goal != 'gripper_none':
            gripper_target_site_id = self.sim.model.site_name2id('final_arm_target')
            gripper_tgt_size = (np.ones(3) * 0.02)
            self.sim.model.site_size[gripper_target_site_id] = gripper_tgt_size
            gripper_tgt_goal = self.goal[0:3] - sites_offset[0]
            # gripper_tgt_goal = self.initial_gripper_xpos[:3] * (np.ones(3) - self.goal_mask[0:3]) + gripper_tgt_goal
            self.sim.model.site_pos[gripper_target_site_id] = gripper_tgt_goal
            obj_goal_start_idx += 3

        for n in range(self.n_objects):
            o_target_site_id = self.sim.model.site_name2id('target{}'.format(n))
            o_tgt_size = (np.ones(3) * 0.02)
            self.sim.model.site_size[o_target_site_id] = o_tgt_size
            o_tgt_goal = self.goal[obj_goal_start_idx:obj_goal_start_idx + 3] - sites_offset[0]
            self.sim.model.site_pos[o_target_site_id] = o_tgt_goal
            obj_goal_start_idx += 3

        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of objects.
        for o in range(self.n_objects):
            oname = 'object{}'.format(o)
            object_xpos = self.initial_gripper_xpos[:2]
            close = True
            while close:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
                close = False
                dist_to_nearest = np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2])
                # Iterate through all previously placed boxes and select closest:
                for o_other in range(o):
                    other_xpos = self.sim.data.get_joint_qpos('object{}:joint'.format(o_other))[:2]
                    dist = np.linalg.norm(object_xpos - other_xpos)
                    dist_to_nearest = min(dist, dist_to_nearest)
                if dist_to_nearest < 0.1:
                    close = True

            object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(oname))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[2] = self.table_height + (self.obj_height / 2) * self.obj_height * 1.05
            self.sim.data.set_joint_qpos('{}:joint'.format(oname), object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        obs = self._get_obs()
        target_goal = None
        if obs is not None:
            if self.gripper_goal != 'gripper_none':
                goal = obs['observation'].copy()[:self.goal_size]
            else:
                goal = obs['observation'].copy()[3:self.goal_size + 3]

            if self.gripper_goal != 'gripper_none' and self.n_objects > 0:
                target_goal_start_idx = 3
            else:
                target_goal_start_idx = 0

            stack_tower = (self.max_tower_height - self.min_tower_height + 1) == self.n_objects

            if not stack_tower:
                if self.n_objects > 0:
                    target_range = self.n_objects
                else:
                    target_range = 1
                for n_o in range(target_range):
                    # too_close = True
                    while True:
                        target_goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range,
                                                                                             self.target_range,
                                                                                             size=3)
                        target_goal += self.target_offset
                        rnd_height = random.randint(self.min_tower_height, self.max_tower_height)
                        self.goal_tower_height = rnd_height
                        target_goal[2] = self.table_height + (rnd_height * self.obj_height) - (self.obj_height / 2)
                        too_close = False
                        for i in range(0, target_goal_start_idx, 3):
                            other_loc = goal[i:i + 3]
                            dist = np.linalg.norm(other_loc[:2] - target_goal[:2], axis=-1)
                            if dist < 0.1:
                                too_close = True
                        if too_close is False:
                            break

                    goal[target_goal_start_idx:target_goal_start_idx + 3] = target_goal.copy()
                    target_goal_start_idx += 3
            else:
                target_goal_xy = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range,
                                                                                        self.target_range,
                                                                                        size=2)
                self.goal_tower_height = self.n_objects
                for n_o in range(self.n_objects):
                    height = n_o + 1
                    target_z = self.table_height + (height * self.obj_height) - (self.obj_height / 2)
                    target_goal = np.concatenate((target_goal_xy, [target_z]))
                    goal[target_goal_start_idx:target_goal_start_idx + 3] = target_goal.copy()
                    target_goal_start_idx += 3

            # Final gripper position
            if self.gripper_goal != 'gripper_none':
                gripper_goal_pos = goal.copy()[-3:]
                if self.gripper_goal == 'gripper_above':
                    gripper_goal_pos[2] += (3 * self.obj_height)
                elif self.gripper_goal == 'gripper_random':
                    too_close = True
                    while too_close:
                        gripper_goal_pos = self.initial_gripper_xpos[:3] + \
                                           self.np_random.uniform(-self.target_range,
                                                                  self.target_range, size=3)
                        gripper_goal_pos[0] += self.random_gripper_goal_pos_offset[0]
                        gripper_goal_pos[1] += self.random_gripper_goal_pos_offset[0]
                        gripper_goal_pos[2] += self.random_gripper_goal_pos_offset[2]
                        if np.linalg.norm(gripper_goal_pos - target_goal, axis=-1) >= 0.1:
                            too_close = False
                goal[:3] = gripper_goal_pos

            return goal.copy()
        else:
            return []

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)

        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # offset the random goal if gripper random is used
        self.random_gripper_goal_pos_offset = (0.25, 0.0, 0.14)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.n_objects > 0:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]