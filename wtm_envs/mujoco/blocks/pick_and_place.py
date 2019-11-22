from gym import utils
from wtm_envs.mujoco import blocks_env


class BlockPickAndPlaceMujocoEnv(blocks_env.BlocksEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', gripper_goal='gripper_none',
                 n_objects=3, min_tower_height=1, max_tower_height=3):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            'object0:joint': [0.1, 0.0, 0.05, 1., 0., 0., 0.],
            'object1:joint': [0.2, 0.0, 0.05, 1., 0., 0., 0.],
            'object2:joint': [0.3, 0.0, 0.05, 1., 0., 0., 0.],
            'object3:joint': [0.4, 0.0, 0.05, 1., 0., 0., 0.],
            'object4:joint': [0.5, 0.0, 0.05, 1., 0., 0., 0.],
        }
        blocks_env.BlocksEnv.__init__(
            self, 'blocks/environment.xml', block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            gripper_goal=gripper_goal, n_objects=n_objects, table_height=0.5, obj_height=0.05,
            min_tower_height=min_tower_height, max_tower_height=n_objects)
        utils.EzPickle.__init__(self)