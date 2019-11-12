from gym import utils
from wtm_envs.mujoco import causal_dep_env

class CausalDependenciesMujocoEnv(causal_dep_env.CausalDependenciesEnv, utils.EzPickle):
    def __init__(self, n_objects=2):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
        }
        causal_dep_env.CausalDependenciesEnv.__init__(
            self, 'causal_dep/environment.xml', n_substeps=20,
            gripper_extra_height=0,
            obj_range=0.2,
            distance_threshold=0.11,
            initial_qpos=initial_qpos,
            n_objects=n_objects, table_height=0.5, obj_height=0.05)
        utils.EzPickle.__init__(self)