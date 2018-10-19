import numpy as np
from wtm_envs.mujoco import keybot_env, utils

import wtm_envs.physical.assets.keybot.motion_controller as robot
class KeybotEnv(keybot_env.KeybotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, gripper_relative_target, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height=None, max_tower_height=None,
            visualize_ik=True
    ):
        self.initial_qpos = initial_qpos
        self.controller = robot.Controller(interval_max=0)
        self.visualize_ik = visualize_ik

        super(KeybotEnv, self).__init__(model_path, n_substeps, gripper_extra_height, block_gripper,
            target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, gripper_relative_target, initial_qpos, reward_type,
            gripper_goal, n_objects, table_height, obj_height, min_tower_height, max_tower_height)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        # Apply action to real environment.
        # action = np.array([0.0, 0.2, 0.2, 0, 0, 0, 0, 0.02401411, 0.02401411])  # REMOVE
        observation = self.controller.ctrl_set_action(action, plot_frame=self.visualize_ik)
        # add the initial base_link positions to the observation keeping it independent from mujoco
        observation = observation + [self.initial_qpos['robot0:slide0'],
                                     self.initial_qpos['robot0:slide1'],
                                     self.initial_qpos['robot0:slide2']]
        grip_pos = observation
        # TODO (fabawi): get the volicity from the motors instead
        grip_velp = [0.01, 0.01, 0.01]
        obs = self._get_obs(grip_pos=grip_pos, grip_velp=grip_velp)

        done = False
        is_success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info = {
            'is_success': is_success
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        return obs, reward, done, info