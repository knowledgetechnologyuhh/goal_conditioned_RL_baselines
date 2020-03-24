import numpy as np
import random

from gym.envs.robotics import rotations
from wtm_envs.mujoco import robot_env, utils
from mujoco_py.generated import const as mj_const
from wtm_envs.mujoco.wtm_env import goal_distance
from wtm_envs.mujoco.wtm_env import WTMEnv

import mujoco_py



class CausalDependenciesEnv(WTMEnv):
    """A simple reacher task with causal dependencies. The robot has to reach n-key locations in order to unlock the goal.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, obj_range,
            distance_threshold, initial_qpos,
            n_objects, table_height, obj_height
    ):
        """Initializes a new Causal Dependencies environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            obj_range (float): range of a uniform distribution for sampling initial object positions
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            n_objects (int): no of objects in the environment. If none, then no_of_objects=0
        """
        self.gripper_extra_height = gripper_extra_height
        self.obj_range = obj_range
        self.distance_threshold = distance_threshold

        self.n_objects = n_objects
        self.keylocs = n_objects - 1
        self.table_height = table_height
        self.obj_height = obj_height
        self.step_ctr = 0
        self.reward_type = 'sparse'

        n_actions = 2
        WTMEnv.__init__(self, model_path=model_path, n_substeps=n_substeps, initial_qpos=initial_qpos,
                        n_actions=n_actions)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        norm_dist = np.linalg.norm(goal_a - goal_b, axis=-1)
        if goal_a.shape[-1] % 2 == 0:
            n_xyz = int(goal_a.shape[-1] / 2)
            max_dist = np.zeros(norm_dist.shape)
            for n in range(n_xyz):
                start = n * 2
                end = start + 2
                subg_a = goal_a[..., start:end]
                subg_b = goal_b[..., start:end]
                dist = np.asarray(np.linalg.norm(subg_a - subg_b, axis=-1))
                if len(max_dist.shape) == 0:
                    max_dist = np.max([float(dist), float(max_dist)])
                else:
                    max_dist = np.max([dist, max_dist], axis=0)

            return max_dist
        else:
            return norm_dist
    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _set_action(self, action):
        assert action.shape == (2,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        pos_ctrl = 0.05 * action  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        action = np.concatenate([pos_ctrl, [0], rot_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        self.step_ctr += 1

    def _get_obs(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep

        #robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        robot_xpos = self.sim.data.get_geom_xpos('robot0:rod').copy()
        robot_xvelp = self.sim.data.get_geom_xvelp('robot0:rod').copy()
        object_pos, object_rot, object_velp, object_velr = (np.array([]) for _ in range(4))

        if self.n_objects > 0:
            for n_o in range(self.n_objects):
                oname = 'object{}'.format(n_o)
                this_object_pos = self.sim.data.get_geom_xpos(oname).copy()

                # remove subgoal if robot is close to it
                if n_o > 0:
                    if np.linalg.norm(robot_xpos[:2] - this_object_pos[:2]) < self.distance_threshold:
                        #this_object_pos[2] -= 0.002
                        object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(n_o)).copy()
                        object_qpos[:3] = this_object_pos
                        swap_pos = self.sim.data.get_joint_qpos('object{}sub:joint'.format(n_o)).copy()
                        self.sim.data.set_joint_qpos('object{}:joint'.format(n_o), swap_pos)
                        self.sim.data.set_joint_qpos('object{}sub:joint'.format(n_o), object_qpos)
                        self.keylocs -= 1

                # rotations
                this_object_rot = rotations.mat2euler(self.sim.data.get_geom_xmat(oname))
                # velocities
                this_object_velp = self.sim.data.get_geom_xvelp(oname) * dt
                this_object_velr = self.sim.data.get_geom_xvelr(oname) * dt

                object_pos = np.concatenate([object_pos, this_object_pos])
                object_rot = np.concatenate([object_rot, this_object_rot])
                object_velp = np.concatenate([object_velp, this_object_velp])
                object_velr = np.concatenate([object_velr, this_object_velr])
        else:
            object_pos = object_rot = object_velp = object_velr = np.array(np.zeros(3))

        #remove cage if all subgoals were reached
        cage_pos = self.sim.data.get_geom_xpos('cage:botback').copy()
        cage_pos[1] = cage_pos[1]-0.05
        if self.keylocs == 0:
            self.sim.data.set_joint_qpos('cage:glassjoint', -1.99)

        obs = np.concatenate([
            robot_xpos, object_pos.ravel(), object_rot.ravel(), cage_pos,
            robot_xvelp, object_velp.ravel(), object_velr.ravel()
        ])

        noisy_obs = self.add_noise(obs.copy(), self.obs_history, self.obs_noise_coefficient)
        achieved_goal = noisy_obs[:2]

        obs = {'observation': noisy_obs.copy(), 'achieved_goal': achieved_goal.copy(), 'desired_goal': self.goal.copy(), 'non_noisy_obs': obs.copy()}

        return obs

    def _reset_sim(self):
        self.step_ctr = 0
        self.sim.set_state(self.initial_state)
        self.keylocs = self.n_objects - 1

        grip_pos = self.initial_gripper_xpos[:2]
        p = [grip_pos]
        for o in range(self.n_objects):
            close = True
            while close:
                close = False
                this_pos = self.sim.data.get_geom_xpos('table0')[:2].copy()
                this_pos += self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
                for pos in p:
                    dist = np.linalg.norm(pos - this_pos)
                    if dist < 0.15:
                        close = True
            p.append(this_pos.copy())

        for o in range(self.n_objects):
            object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(o))
            assert object_qpos.shape == (7,)
            object_qpos[:2] = p[o+1]
            object_qpos[2] = self.table_height + (self.obj_height / 2)
            self.sim.data.set_joint_qpos('object{}:joint'.format(o), object_qpos)

        #set cage position
        goal_pos = self.sim.data.get_joint_qpos('object0:joint').copy()
        goal_pos[2] = goal_pos[2]+0.04
        self.sim.data.set_joint_qpos('cage:joint', goal_pos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        obs = self._get_obs()
        if obs is not None:
            goal = obs['observation'].copy()[3:5]
        return goal

    def _obs2goal(self, obs):
        g = obs[:2].copy()
        return g

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_geom_xpos('robot0:rod').copy()
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_geom_xpos('robot0:rod').copy()

    def get_scale_and_offset_for_normalized_subgoal(self):
        offset = self.sim.data.get_geom_xpos('table0')[:2].copy()
        scale_xy = self.obj_range
        scale = np.array([scale_xy, scale_xy])
        return scale, offset
