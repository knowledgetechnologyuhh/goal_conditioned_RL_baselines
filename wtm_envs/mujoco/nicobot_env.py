import numpy as np
from wtm_envs.mujoco import blocks_env, utils

class NicobotEnv(blocks_env.BlocksEnv):
    """Superclass for all Nicobot environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, gripper_relative_target, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height=None, max_tower_height=None,
    ):
        self.gripper_relative_target = gripper_relative_target
        super(NicobotEnv, self).__init__(model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height, max_tower_height)


    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        if self.gripper_relative_target:
            pos_ctrl *= 0.02  # limit maximum change in position. was 0.05\
            ref_frame = None
        else:  # Absolute target relative to the robot frame
            pos_ctrl[0] *= 0.1  # limit maximum change in position. was 0.05
            pos_ctrl[1] *= 0.15  # limit maximum change in position. was 0.05
            pos_ctrl[2] *= 0.10  # limit maximum change in position. was 0.05
            pos_ctrl[0] += 0.30  # add constant to x-axis to avoid generating actions behind the robot
            pos_ctrl[1] += 0.05
            pos_ctrl[2] += 0.6
            ref_frame = self.sim.data.get_body_xpos('base_link')

        # pos_ctrl[0] += 0.20  # add constant to x-axis to avoid generating actions behind the robot

        rot_ctrl = [0., 0., 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion # [1., -1., -1, 0]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (3,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action, absolute_ref=ref_frame)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.

        gripper_rotation = np.array([0.32600718, 0.86378862, -0.2913803, 0.25037185])
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

        gripper_target = np.array([0.40, -0.1, 0.2 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        x_start = gripper_target[0]
        x_end = gripper_target[0] - 0.2
        steps = 10
        step = (x_start - x_end) / steps
        current_x = x_start
        while current_x > x_end:
            gripper_target[0] = current_x
            self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
            current_x -= step
            self.sim.step()
            # self.render()
            # print(current_x)


        # Offset the random goal if gripper random is used
        if self.gripper_relative_target:
            self.random_gripper_goal_pos_offset = (0., 0., 0.)  # (0.23, 0.0, 0.0)
        else:
            self.random_gripper_goal_pos_offset = (0., 0., 0.)  # (0.13, 0.0, 0.0)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.n_objects > 0:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
