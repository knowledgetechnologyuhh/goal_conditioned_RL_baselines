from gym import utils
from wtm_envs.mujoco import ant_env
from wtm_envs.mujoco.hook_env_pddl import PDDLHookEnv
import numpy as np

class AntReacherEnv(ant_env.AntEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):

        # Provide initial state space consisting of the ranges for all joint angles and velocities.
        # In the Ant Reacher task, we use a random initial torso position and use fixed values for the remainder.
        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos), 1))
        initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos), 1)
        initial_joint_ranges[0] = np.array([-9.5, 9.5])
        initial_joint_ranges[1] = np.array([-9.5, 9.5])

        # Concatenate velocity ranges
        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

        max_range = 9.5
        goal_space_train = [[-max_range, max_range], [-max_range, max_range], [0.45, 0.55]]
        goal_space_test = [[-max_range, max_range], [-max_range, max_range], [0.45, 0.55]]

        # Provide a function that maps from the state space to the end goal space.  This is used to (i) determine whether the agent should be given the sparse reward and (ii) for Hindsight Experience Replay to determine which end goal was achieved after a sequence of actions.
        project_state_to_end_goal = lambda sim, state: state[:3]

        # Set end goal achievement thresholds.  If the agent is within the threshold for each dimension, the end goal has been achieved and the reward of 0 is granted.

        # For the Ant Reacher task, the end goal will be the desired (x,y) position of the torso
        len_threshold = 0.5
        height_threshold = 0.2
        end_goal_thresholds = np.array([len_threshold, len_threshold, height_threshold])


        # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.

        # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
        cage_max_dim = 11.75
        max_height = 1
        max_velo = 3
        subgoal_bounds = np.array([[-cage_max_dim,cage_max_dim],[-cage_max_dim,cage_max_dim],[0,max_height],[-max_velo, max_velo],[-max_velo, max_velo]])


        # Provide state to subgoal projection function.
        # a = np.concatenate((sim.data.qpos[:2], np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in range(3)])))
        project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))


        # Set subgoal achievement thresholds
        velo_threshold = 0.5
        quat_threshold = 0.5
        # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
        subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])



        name = "ant_reacher.xml"
        ant_env.AntEnv.__init__(
            self, 'ant_reacher/environment.xml', n_substeps=15,
            reward_type=reward_type, name=name, goal_space_train=goal_space_train, goal_space_test=goal_space_test,
            project_state_to_end_goal=project_state_to_end_goal, project_state_to_subgoal=project_state_to_subgoal,
            end_goal_thresholds=end_goal_thresholds, initial_state_space=initial_state_space,
            subgoal_bounds=subgoal_bounds, subgoal_thresholds=subgoal_thresholds)
        utils.EzPickle.__init__(self)
