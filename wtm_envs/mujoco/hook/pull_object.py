from gym import utils
from wtm_envs.mujoco import hook_env
from wtm_envs.mujoco.hook_env_pddl import RakeObjectThresholds

class HookMujocoEnv(hook_env.HookEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', gripper_goal='gripper_none',
                 n_objects=2, min_tower_height=1, max_tower_height=3,
                 easy=1):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            'object0:joint': [0.1, 0.0, 0.05,  -0.9908659, 0, 0, -0.1348509]
        }
        hook_env.HookEnv.__init__(
            self, 'hook/environment.xml', block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,    # 0.15 target_range is used to define the target position wrt current object position
            # distance_threshold=0.02,
            distance_threshold=RakeObjectThresholds.distance_threshold,  #0.02
            initial_qpos=initial_qpos, reward_type=reward_type,
            gripper_goal=gripper_goal, n_objects=n_objects, table_height=0.5, obj_height=0.05,
            min_tower_height=min_tower_height, max_tower_height=max_tower_height,
            easy=easy)
        utils.EzPickle.__init__(self)