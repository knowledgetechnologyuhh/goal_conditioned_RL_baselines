from gym import utils
from wtm_envs.mujoco import ur5_env
import numpy as np

class Ur5ReacherEnv(ur5_env.UR5Env, utils.EzPickle):
    def __init__(self, reward_type='sparse', obs_type=1):
        # Provide initial state space consisting of the ranges for all joint angles and velocities. In the UR5
        # Reacher task we use a random initial shoulder position and use fixed values for the remainder.  Initial
        # joint velocities are set to 0.

        initial_joint_pos = np.array([5.96625837e-03, -np.pi/8. + 3.22757851e-03, -1.27944547e-01])
        initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos), 1))
        initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos), 1)
        # initial_joint_ranges[0] = np.array([-np.pi / 8, np.pi / 8])
        initial_joint_ranges[0] = np.array([-np.pi / 2, np.pi / 2])
        initial_joint_ranges[1] = np.array([-np.pi / 8, np.pi / 8])
        initial_joint_ranges[2] = np.array([-np.pi / 4, np.pi / 4])

        initial_state_space = np.concatenate((initial_joint_ranges, np.zeros((len(initial_joint_ranges), 2))), 0)

        # Provide end goal space.  The code supports two types of end goal spaces if user would like to train on a
        # larger end goal space.  If user needs to make additional customizations to the end goals,
        # the "get_next_goal" method in "environment.py" can be updated.

        # In the UR5 reacher environment, the end goal will be the desired joint positions for the 3 main joints.
        goal_space_train = [[-np.pi, np.pi], [-np.pi / 4, 0], [-np.pi / 4, np.pi / 4]]
        goal_space_test = [[-np.pi, np.pi], [-np.pi / 4, 0], [-np.pi / 4, np.pi / 4]]

        # Provide a function that maps from the state space to the end goal space.  This is used to determine whether
        # the agent should be given the sparse reward.  It is also used for Hindsight Experience Replay to determine
        # which end goal was achieved after a sequence of actions.

        # Supplementary function that will ensure all angles are between [-2*np.pi,2*np.pi]
        def bound_angle(angle):
            bounded_angle = np.absolute(angle) % (2 * np.pi)
            if angle < 0:
                bounded_angle = -bounded_angle

            return bounded_angle

        project_state_to_end_goal = \
            lambda sim, state: np.array([bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))])

        # Set end goal achievement thresholds.  If the agent is within the threshold for each dimension, the end goal
        # has been achieved and the reward of 0 is granted.
        angle_threshold = np.deg2rad(10)
        end_goal_thresholds = np.array([angle_threshold, angle_threshold, angle_threshold])

        # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal
        # space can be the same as the state space or some other projection out of the state space.  In our
        # implementation of the UR5 reacher task, the subgoal space is the state space, which is the concatenation of
        # all joint positions and joint velocities.

        subgoal_bounds = np.array(
            [[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi], [-4, 4], [-4, 4], [-4, 4]])

        # Provide state to subgoal projection function.
        project_state_to_subgoal = \
            lambda sim, state: np.concatenate(
                (np.array([bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))]),
                 np.array([4 if sim.data.qvel[i] > 4
                           else -4
                 if sim.data.qvel[i] < -4
                 else sim.data.qvel[i] for i in range(len(sim.data.qvel))])))

        # Set subgoal achievement thresholds
        velo_threshold = 2
        subgoal_thresholds = \
            np.concatenate(
                (np.array([angle_threshold for i in range(3)]), np.array([velo_threshold for i in range(3)])))
        # initial_qpos = {
        #     'robot0:slide0': 0.0,
        #     'robot0:slide1': 0.0,
        #     'robot0:slide2': 0.0,
        #     'object0:joint': [0.1, 0.0, 0.05,  -0.9908659, 0, 0, -0.1348509]
        # }

        name = "ur5.xml"

        ur5_env.UR5Env.__init__(
            self, 'ur5/ur5.xml', n_substeps=20,
            reward_type=reward_type, name=name, goal_space_train=goal_space_train, goal_space_test=goal_space_test,
            project_state_to_end_goal=project_state_to_end_goal, project_state_to_subgoal=project_state_to_subgoal,
            end_goal_thresholds=end_goal_thresholds, initial_state_space=initial_state_space,
            initial_joint_pos=initial_joint_pos,
            subgoal_bounds=subgoal_bounds, subgoal_thresholds=subgoal_thresholds, obs_type=obs_type
        )
        utils.EzPickle.__init__(self)