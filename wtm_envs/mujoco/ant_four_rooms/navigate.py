from gym import utils
from wtm_envs.mujoco import ant_env
from wtm_envs.mujoco.hook_env_pddl import PDDLHookEnv

class AntFourRoomsEnv(ant_env.AntEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):

        name = "ant_four_rooms.xml"
        ant_env.AntEnv.__init__(
            self, 'ant_four_rooms/environment.xml', n_substeps=15,
            reward_type=reward_type, name=name)
        utils.EzPickle.__init__(self)
