from gym import utils
from wtm_envs.mujoco import fetch_env


class FetchBuildTowerEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', gripper_goal='gripper_none',
                 n_objects=3, min_tower_height=1, max_tower_height=3):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 1.33, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 1.03, 0.4, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.73, 0.4, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.43, 0.4, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.13, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/build_tower.xml', block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            gripper_goal=gripper_goal, n_objects=n_objects, table_height=0.5, obj_height=0.05,
            min_tower_height=min_tower_height, max_tower_height=max_tower_height)
        utils.EzPickle.__init__(self)