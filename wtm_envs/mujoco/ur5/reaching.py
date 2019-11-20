from gym import utils
from wtm_envs.mujoco import ur5_env
from wtm_envs.mujoco.ur5_env_pddl import PDDLUR5Env

class Ur5MujocoEnv(ur5_env.UR5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse', gripper_goal='gripper_none',
                 n_objects=2, min_tower_height=1, max_tower_height=3,
                 easy=1):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            'object0:joint': [0.1, 0.0, 0.05,  -0.9908659, 0, 0, -0.1348509]
        }
        ur5_env.UR5Env.__init__(
            self, 'ur5/ur5.xml', block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,    # 0.15 target_range is used to define the target position wrt current object position
            # distance_threshold=0.02,
            distance_threshold=PDDLUR5Env.distance_threshold,  #0.02
            initial_qpos=initial_qpos, reward_type=reward_type,
            gripper_goal=gripper_goal, n_objects=n_objects, table_height=PDDLUR5Env.table_height,
            obj_height=PDDLUR5Env.obj_height,
            min_tower_height=min_tower_height, max_tower_height=n_objects,
            easy=easy)
        utils.EzPickle.__init__(self)