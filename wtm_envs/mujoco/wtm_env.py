import numpy as np
from wtm_envs.mujoco import robot_env, utils
import mujoco_py
from queue import deque
from mujoco_py import modder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    norm_dist = np.linalg.norm(goal_a - goal_b, axis=-1)
    if goal_a.shape[-1] % 3 == 0:
        n_xyz = int(goal_a.shape[-1] / 3)
        max_dist = np.zeros(norm_dist.shape)
        for n in range(n_xyz):
            start = n * 3
            end = start + 3
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

class PercDeque(deque):
    def __init__(self, maxlen, perc_recomp=100):
        self.ctr = 0
        self.upper_perc = None
        self.lower_perc = None
        self.perc_recomp = perc_recomp
        super(PercDeque, self).__init__(maxlen=maxlen)

    def append(self, vec):
        super(PercDeque, self).append(vec)
        if self.ctr == self.maxlen:
            self.ctr = 0
        if self.ctr % self.perc_recomp == 0:
            hist_vec = np.array(self)
            self.upper_perc = np.percentile(hist_vec, 75, axis=0)
            self.lower_perc = np.percentile(hist_vec, 25, axis=0)
        self.ctr += 1


class WTMEnv(robot_env.RobotEnv):
    def __init__(
        self, model_path, n_substeps, initial_qpos, n_actions=4
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            gripper_goal ('gripper_none', 'gripper_above', 'gripper_random'): the gripper's goal location
            n_objects (int): no of objects in the environment. If none, then no_of_objects=0
            min_tower_height (int): the minimum height of the tower.
            max_tower_height (int): the maximum height of the tower.
        """

        self._viewers = {}

        self.obs_history = PercDeque(maxlen=5000)
        self.obs_noise_coefficient = 0.0

        self.plan_cache = {}
        self.goal_hierarchy = {}
        self.goal = []
        self.final_goal = []
        self.graph_values = {}

        super(WTMEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

        # self.mod = modder.TextureModder(self.sim)

        # assert self.gripper_goal in ['gripper_above', 'gripper_random'], "gripper_none is not supported anymore"

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal)
            return (success - 1).astype(np.float32)
        else:
            d = goal_distance(achieved_goal, goal)
            return -d

    # RobotEnv methods
    # ----------------------------
    def _step_callback(self):
        if "block_gripper" in self.__dict__.keys():
            if self.block_gripper:
                self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
                self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
                self.sim.forward()

    def _goal2obs(self, goal):
        if len(goal.shape) == 1:
            goal_arr = np.array([goal])
        else:
            goal_arr = goal
        assert len(goal_arr.shape) == 2
        obs = []
        o_dims = self.observation_space.spaces['observation'].shape[0]
        o = np.zeros(o_dims, np.float32)
        for g in goal_arr:
            o[:self.goal_size] = g
            obs.append(o.copy())
        obs = np.array(obs)
        if len(goal.shape) == 1:
            return obs[0]
        else:
            return obs

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        self.step_ctr += 1

    def add_noise(self, vec, history, noise_coeff):
        history.append(vec)
        range = history.upper_perc - history.lower_perc
        coeff_range = noise_coeff * range
        noise = np.random.normal(loc=np.zeros_like(coeff_range), scale=coeff_range)
        vec = vec.copy() + noise
        return vec

    def _get_viewer(self, mode='human'):
        viewer = self._viewers.get(mode)
        if viewer is None:
            if mode == 'human':
                viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                viewer = mujoco_py.MjViewer(self.sim)
                # The following should work but it does not. Therefore, replaced by human rendering (with MjViewer, the line above) now.
                # viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                # viewer = mujoco_py.MjRenderContext(self.sim, -1)
            self._viewers[mode] = viewer
            self._viewer_setup(mode=mode)

        return self._viewers[mode]

    def _viewer_setup(self, mode='human'):
        if mode == 'human':
            body_id = self.sim.model.body_name2id('robot0:gripper_link')
            lookat = self.sim.data.body_xpos[body_id]
            for idx, value in enumerate(lookat):
                self._viewers[mode].cam.lookat[idx] = value
            self._viewers[mode].cam.distance = 2.5
            self._viewers[mode].cam.azimuth = 132.
            self._viewers[mode].cam.elevation = -14.
        elif mode == 'rgb_array':
            body_id = self.sim.model.body_name2id('robot0:gripper_link')
            lookat = self.sim.data.body_xpos[body_id]
            for idx, value in enumerate(lookat):
                self._viewers[mode].cam.lookat[idx] = value
            self._viewers[mode].cam.distance = 1.
            self._viewers[mode].cam.azimuth = 180.
            self._viewers[mode].cam.elevation = -40.

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render()
            # window size used for old mujoco-py:
            width, height = 1920, 1180
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()
        if bool(self.graph_values):
            body_names = [self.sim.model.body_id2name(x) for x in np.arange(self.sim.model.nbody)]
            if 'graph_body' in body_names:  # check if canvas in XML
                self._get_viewer().vopt.geomgroup[3] = 1  # make canvas visible
                self.mod.set_rgb("graph_geom", self.create_graph())

    def create_graph(self):
        # create Graph
        fig = plt.figure(figsize=(6.4, 6.4))
        canvas = FigureCanvasAgg(fig)
        keys = self.graph_values.keys()
        keys = filter(lambda x: x[-2:] != '_x', keys)
        for i, key in enumerate(keys):
            frame_on = i==0
            ax = fig.add_subplot(111, label=str(i), frame_on=frame_on)
            ax.set_ylabel(str(key), color="C"+str(i))
            ax.set_xlabel('step', color="C"+str(i))
            if i % 2 != 0:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
            ax.tick_params(axis='y', colors="C"+str(i))
            ax.plot(self.graph_values[key+'_x'], self.graph_values[key], color="C"+str(i))
        plt.tight_layout()
        canvas.draw()

        # convert to rgb array
        buf = canvas.tostring_rgb()
        ncols, nrows = canvas.get_width_height()
        plt.close(fig)
        return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

    def add_graph_values(self, axis_name, val, x, reset=False):
        if reset and axis_name in self.graph_values.keys():
            del self.graph_values[axis_name]
            del self.graph_values[axis_name+'_x']
        if axis_name in self.graph_values:
            self.graph_values[axis_name].append(val)
            self.graph_values[axis_name+'_x'].append(x)
        else:
            self.graph_values[axis_name]=[val[0]]
            self.graph_values[axis_name+'_x'] = [x]