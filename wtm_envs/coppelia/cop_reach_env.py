from pyrep.pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import IKError
import numpy as np
from os.path import dirname, join, abspath
import gym
from gym import spaces
from gym.utils import seeding


SCENE_FILE = join(dirname(abspath(__file__)),
                  'CopReacherEnv.ttt')

class ManipulatorPro (Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'mp', num_joints=6)

POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]

class ReacherEnv(gym.GoalEnv):
    """
    Environment with Reacher tasks that uses CoppeliaSim and the Franka Emika Panda robot.
    Args:
        render: If render=0, CoppeliaSim will run in headless mode.
        tmp: whether the environment is only used temporarily to acquire e.g. the shape of the observation space.
        ik: whether to use inverse kinematics. If not, the actuators will be controlled directly.
            Note, that also the observation changes, when ik is set.
            !!!The IK can not always be computed. In that case, the action is not carried out!!!
    """
    def __init__(self, render=1, tmp=False, ik=1):
        print('\033[92m' + 'Creating new Env' + '\033[0m')
        render = bool(render)
        self.ik = bool(ik)

        # PyRep initialization
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=not render)
        self.pr.start()

        # load robot and set position
        self.agent = ManipulatorPro()
        if not self.ik:
            self.agent.set_control_loop_enabled(False)
            self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.vis = {}
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        # define goal
        self.goal = self._sample_goal()
        self.goal_hierarchy = {}  # for herhrl

        # set action space
        if self.ik:
            self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        else:
            self.action_space = spaces.Box(-1., 1., shape=(6,), dtype='float32')
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),))

        # set if environment is only for short usage
        self.tmp = tmp

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        achieved_goal = self.agent_ee_tip.get_position()
        if self.ik:
            obs = achieved_goal
        else:
            obs = np.concatenate([achieved_goal,
                                  self.agent.get_joint_positions(),
                                  self.agent.get_joint_velocities()])

        obs = {'observation': obs.copy(), 'achieved_goal': achieved_goal.copy(),
               'desired_goal': np.array(self.goal.copy()),
               'non_noisy_obs': obs.copy()}
        return obs

    def _sample_goal(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        return pos

    def reset(self):
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.goal = self._sample_goal()
        obs = self._get_obs()
        if self.tmp > 0:
            self.tmp -= 1
            print('\033[91m' + 'This Env will shut down after ' + str(self.tmp) + ' resets' + '\033[0m')
            if self.tmp == 0:
                self.close()
        return obs

    def compute_reward(self, achieved_goal, goal, info):
        if achieved_goal.shape[0] != 3:
            reward = [self.compute_reward(g1, g2, info) for g1, g2 in zip(achieved_goal, goal)]
        else:
            ax, ay, az = achieved_goal  # self.agent_ee_tip.get_position()
            tx, ty, tz = goal  # self.target.get_position()
            dist = np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)
            if dist < 0.05:
                reward = 0
            else:
                reward = -1
        return np.array(reward)

    def _is_success(self, achieved_goal, desired_goal):
        return self.compute_reward(achieved_goal, desired_goal, {}) == 0

    def _set_action(self, action):
        if self.ik:
            pos = self.agent_ee_tip.get_position().copy()
            quat = self.agent_ee_tip.get_quaternion().copy()
            pos += (action * 0.05)
            try:
                new_joint_angles = self.agent.solve_ik(pos, quaternion=quat)
                self.agent.set_joint_target_positions(new_joint_angles)
            except IKError:
                print('Attempting to reach out of reach')
        else:
            self.agent.set_joint_target_velocities(action)

    def step(self, action):
        self._set_action(action)
        self.pr.step()  # Step the physics simulation
        done = False
        obs = self._get_obs()
        is_success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info = {'is_success': is_success}

        r = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        return obs, r, done, info

    def get_scale_and_offset_for_normalized_subgoal(self):  # for herhrl
        # let a = {-1, ..., 1}
        # then: offset + a * scale = all possible target positions
        scale = (np.array(POS_MAX) - np.array(POS_MIN)) / 2
        offset = (np.array(POS_MAX) + np.array(POS_MIN)) / 2
        return scale, offset

    def add_graph_values(self, axis_name, val, x, reset=False):  # for herhrl
        pass  # not implemented yet

    def _obs2goal(self, obs):
        if len(obs.shape) == 1:
            obs_arr = np.array([obs])
        else:
            obs_arr = obs
        assert len(obs_arr.shape) == 2
        goals = []
        for o in obs_arr:
            goals.append(o[:3])
        goals = np.array(goals)
        if len(obs.shape) == 1:
            return goals[0]
        else:
            return goals

    def visualize(self, names_pos_col={}):
        """
        Takes a dictionary with names, positions and colors that is structured as follows:
        {'name': {'pos': [0.8, -0.1, 1.1], 'col': [.0, .9, .0]}, 'name2': {'pos': [1.0, 0.1, 1.3], 'col': [.0, .0, .9]}}
        Then cubes with the name are created in the specified color and moved to the position.
        """
        for name in names_pos_col:
            if name not in self.vis:
                self.vis[name] = Shape.create(PrimitiveShape.CUBOID, [0.04]*3, mass=0, respondable=False, static=True,
                                              position=names_pos_col[name]['pos'], color=names_pos_col[name]['col'])
            else:
                self.vis[name].set_position(names_pos_col[name]['pos'])
                self.vis[name].set_color(names_pos_col[name]['col'])

    def close(self):
        print('\033[91m' + 'Closing Env' + '\033[0m')
        if self.ik:
            print('\033[91m' + 'You are running with inverse kinematics. Sometimes the IK are not working and the '
                               'action can not be carried out. In that case, \'Attempting to reach out of reach\' is '
                               'printed.' + '\033[0m')
        self.pr.stop()
        self.pr.shutdown()
