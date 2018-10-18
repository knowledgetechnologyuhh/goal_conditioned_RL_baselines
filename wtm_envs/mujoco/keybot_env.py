import numpy as np
from wtm_envs.mujoco import tower_env, utils

class KeybotEnv(tower_env.TowerEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height=None, max_tower_height=None,
    ):
        super(KeybotEnv, self).__init__(model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height, max_tower_height)


    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position. was 0.05
        # pos_ctrl[0] += 0.20  # add constant to x-axis to avoid generating actions behind the robot

        rot_ctrl = [1., 0., 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([0.0, 0.0, 0.0 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_rotation = np.array([0., 0., 0., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Offset the random goal if gripper random is used
        self.random_gripper_goal_pos_offset = (1.0, 0.0, 0.0)

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.n_objects > 0:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
