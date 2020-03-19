from pyrep.pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
import numpy as np

SCENE_FILE = 'scene_reinforcement_learning_env.ttt'
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 5
EPISODE_LENGTH = 200

class ReacherEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = Panda()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

    def _get_obs(self):
        # Return state containing arm joint angles/velocities & target position
        obs = np.concatenate([self.agent.get_joint_positions(),
                              self.agent.get_joint_velocities()])
        achieved_goal = self.agent_ee_tip.get_position()
        obs = {'observation': obs.copy(), 'achieved_goal': achieved_goal.copy(),
               'desired_goal': self.goal.copy(),
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
        return self._get_obs()

    def compute_reward(self):
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

    def _set_action(self):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm

    def _is_success(self):
        return self.compute_reward() > -0.05

    def step(self, action):
        self._set_action(action)
        self.pr.step()  # Step the physics simulation
        done = False
        is_success = self._is_success()
        info = {'is_success': is_success}
        return self._get_obs(), self.compute_reward(), done, info

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass

env = ReacherEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        reward, next_state = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
