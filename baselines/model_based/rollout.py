import matplotlib
# matplotlib.use('Agg')

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mujoco_py import MujocoException
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from collections import deque
from baselines.template.util import convert_episode_to_batch_major, store_args
import matplotlib.pyplot as plt
from mujoco_py.generated import const as mj_const
from plot.plot_model_train import plot_model_train
from multiprocessing import Process, Queue
import imageio


class RolloutWorker(Rollout):
    @store_args
    def __init__(self, make_env, policy, dims, logger, T, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.
        """
        Rollout.__init__(self, make_env, policy, dims, logger, T, **kwargs)

        self.pred_err = None
        self.pred_accumulated_err = None
        self.pred_err_std = None
        self.pred_steps = None
        self.mj_pred_err = None
        self.mj_pred_accumulated_err = None
        self.mj_pred_err_std = None
        self.mj_pred_steps = None
        self.loss_pred_steps = None
        self.loss_histories = []
        self.buff_obs_variance = None
        self.surprise_fig = None
        self.replayed_episodes = []
        self.top_exp_replay_values = deque(maxlen=10)
        self.episodes_per_epoch = None
        self.visualize_replay = False
        self.record_replay = True
        self.do_plot = False
        self.test_mujoco_err = True
        self.acc_err_steps = 5 # Number of steps to accumulate the prediction error

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        for i,l in enumerate(self.loss_histories):
            loss_key = 'loss-{}'.format(i)
            loss_key_map = ['observation loss', 'loss prediction loss']
            if i < len(loss_key_map):
                loss_key = loss_key_map[i]
            if len(l) > 0:
                logs += [(loss_key, l[-1])]
            # if len(l) > 1:
            #     loss_grad = l[-1] / l[-2]
            # else:
            #     loss_grad = np.nan
            # logs += [('{} grad'.format(loss_key), loss_grad)]

        if self.pred_err is not None :
            logs += [('pred_err', self.pred_err)]
        if self.pred_steps is not None:
            logs += [('pred_steps', self.pred_steps)]
        if self.pred_accumulated_err is not None:
            logs += [('acc. err', self.pred_accumulated_err)]
        if self.loss_pred_steps is not None:
            logs += [('loss_pred_steps', self.loss_pred_steps)]
        if self.mj_pred_err is not None :
            logs += [('mj_pred_err', self.mj_pred_err)]
        if self.mj_pred_steps is not None:
            logs += [('mj_pred_steps', self.mj_pred_steps)]
        if self.mj_pred_accumulated_err is not None:
            logs += [('mj acc. err', self.mj_pred_accumulated_err)]
        if self.buff_obs_variance is not None:
            logs += [('buffer observation variance', self.buff_obs_variance)]
        if self.buff_obs_variance is not None and self.pred_accumulated_err is not None:
            logs += [('variance_div_acc_err', self.buff_obs_variance / self.pred_accumulated_err)]
        logs += [('episode', self.n_episodes)]

        return logger(logs, prefix)

    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        avg_epoch_losses = []
        last_stored_idxs = []
        self.episodes_per_epoch = n_episodes
        for episode in range(n_episodes):
            print("episode {} / {}".format(episode, n_episodes))
            ro_start = time.time()
            episode, mj_states = self.generate_rollouts(return_states=True)
            stored_idxs = self.policy.store_episode(episode, mj_states=mj_states)
            last_stored_idxs = stored_idxs
            # Remove old rollouts from already replayed episodes.
            for i in stored_idxs:
                if i in self.replayed_episodes:
                    self.replayed_episodes.remove(i)

            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_train_batches):
                losses = self.policy.train()
                if not isinstance(losses, tuple):
                    losses = [losses]

                if self.loss_histories == []:
                    self.loss_histories = [deque(maxlen=2) for _ in losses]
                if avg_epoch_losses == []:
                    avg_epoch_losses = [0 for _ in losses]
                for idx, loss in enumerate(losses):
                    avg_epoch_losses[idx] += loss
            dur_train += time.time() - train_start
        for idx,loss in enumerate(avg_epoch_losses):
            avg_loss = (loss / n_episodes) / n_train_batches
            self.loss_histories[idx].append(avg_loss)
        # Update learning rate if loss gradient exceeds 1 for any loss history:
        if self.adaptive_model_lr:
            for l in self.loss_histories:
                if len(l) >= 2:
                    grad = l[-1] / l[-2]
                    if grad > 1.1:
                        self.policy.model_lr *= 0.95

        self.buff_obs_variance = self.policy.model_replay_buffer.get_variance(column='o')
        self.test_prediction_error(last_stored_idxs)
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        self.replay_experience()
        if self.do_plot:
            plot_model_train("/".join(self.logger.get_dir().split("/")[:-1]))
        return updated_policy, time_durations


    def test_prediction_error(self, buffer_idxs):

        this_pred_err_hist = []
        this_accumulated_pred_err = []
        this_pred_std_hist = []
        this_mj_pred_err_hist = []
        this_mj_accumulated_pred_err = []
        this_mj_pred_std_hist = []


        loss_pred_threshold_perc = 20 # Loss prediction threshold in %

        batch, idxs = self.policy.sample_batch(idxs=buffer_idxs)
        s = None

        o_s_batch, o2_s_batch, u_s_batch, loss_s_batch, loss_pred_s_batch = batch[0], batch[1], batch[2], batch[3], batch[4]

        # Measure mean prediction error
        for step in range(self.T):
            o_s_step, o2_s_step, u_s_step = \
                o_s_batch[:, step:step + 1, :], o2_s_batch[:, step:step + 1, :], \
                u_s_batch[:, step:step + 1, :]
            o2_pred, l_s, s = self.policy.forward_step(u_s_step, o_s_step, s)
            err = np.abs(o2_s_step - o2_pred)
            ep_err_mean_step = np.mean(err,axis=2)
            ep_err_mean = np.mean(ep_err_mean_step)
            ep_err_std = np.std(err)
            this_pred_err_hist.append(ep_err_mean)
            this_pred_std_hist.append(ep_err_std)

        # Test mujoco prediction error as baseline
        if self.test_mujoco_err:
            for step in range(self.T):
                o_s_step, o2_s_step, u_s_step = \
                    o_s_batch[:, step, :], o2_s_batch[:, step, :], \
                    u_s_batch[:, step, :]
                assert len(self.envs) == len(buffer_idxs) # Number of parallel rollout environment instances must be equal to to batch size.
                mj_states = [self.policy.model_replay_buffer.mj_states[buff_idx][step] for buff_idx in buffer_idxs]
                [self.envs[i].env.sim.set_state(mj_state) for i, mj_state in enumerate(mj_states)]
                [self.envs[i].env.sim.forward() for i, mj_state in enumerate(mj_states)]
                o_pred = np.array([self.envs[i].env._get_obs()['observation'] for i in range(len(buffer_idxs))])

                # TODO: For debugging... tmp_err should be 0 or very close to 0, as we are just reloading the mujoco state that corresponds to the observation.
                # TODO: However, tt is larger than expected. Specifically, it is larger that the accumulated mujoco error computed below.
                # TODO: This should not be the case.
                tmp_err = np.abs(o_s_step - o_pred)
                tmp_err_mean = np.mean(tmp_err)

                [self.envs[i].env.step(u) for i, u in enumerate(u_s_step)]
                o2_pred = np.array([self.envs[i].env._get_obs()['observation'] for i in range(len(buffer_idxs))])
                err = np.abs(o2_s_step - o2_pred)
                ep_err_mean = np.mean(err)
                ep_err_std = np.std(err)
                this_mj_pred_err_hist.append(ep_err_mean)
                this_mj_pred_std_hist.append(ep_err_std)


        # Measure the number of steps that can be forward propagated until the observation error gets larger than the goal achievement threshold.
        initial_o_s = o_s_batch[:,0:1,:]
        s = None
        this_pred_steps = [self.T for _ in initial_o_s]
        this_mj_pred_steps = [self.T for _ in initial_o_s]
        this_loss_pred_steps = [self.T for _ in initial_o_s]
        o_s_step = initial_o_s
        for step in range(self.T):
            o2_s_step, u_s_step = o2_s_batch[:, step:step + 1, :], u_s_batch[:, step:step + 1, :]
            o_s_step, l_s, s = self.policy.forward_step(u_s_step, o_s_step, s)
            flattened_o_s_step = o_s_step[:, 0,:]
            flattened_o2_s_step = o2_s_step[:, 0, :]
            fwd_goal_g = self.policy.env._obs2goal(flattened_o_s_step)
            orig_g = self.policy.env._obs2goal(flattened_o2_s_step)
            pred_successes = self.policy.env._is_success(fwd_goal_g, orig_g)
            for batch_idx, pred_success in enumerate(pred_successes):
                if not pred_success and this_pred_steps[batch_idx] == self.T:
                    this_pred_steps[batch_idx] = step
                loss_pred_ok = loss_s_batch[batch_idx][step] + (loss_s_batch[batch_idx][step] * loss_pred_threshold_perc / 100) < l_s[batch_idx][0]
                if not loss_pred_ok and this_loss_pred_steps[batch_idx] == self.T:
                    this_loss_pred_steps[batch_idx] = step
            if step == self.acc_err_steps:
                this_accumulated_pred_err = np.abs(flattened_o_s_step - flattened_o2_s_step)

        # TODO: The accumulated mujoco error is smaller than the per-step mujoco error computed above. Also,
        # TODO: Mujoco seems to always be able to reproduce all steps. This should also not be the case.
        if self.test_mujoco_err:
            assert len(self.envs) == len(buffer_idxs)  # Number of parallel rollout environment instances must be equal to to batch size.
            init_mj_states = [self.policy.model_replay_buffer.mj_states[buff_idx][0] for buff_idx in buffer_idxs]
            [self.envs[i].env.sim.set_state(mj_state) for i, mj_state in enumerate(init_mj_states)]
            [self.envs[i].env.sim.forward() for i, mj_state in enumerate(mj_states)]
            for step in range(self.T):
                o2_s_step, u_s_step = o2_s_batch[:, step, :], u_s_batch[:, step, :]
                [self.envs[i].env.step(u) for i, u in enumerate(u_s_step)]
                o_s_step = np.array([self.envs[i].env._get_obs()['observation'] for i in range(len(buffer_idxs))])
                fwd_goal_g = self.policy.env._obs2goal(o_s_step)
                orig_g = self.policy.env._obs2goal(o2_s_step)
                pred_successes = self.policy.env._is_success(fwd_goal_g, orig_g)
                err = np.abs(o_s_step - o2_s_step)
                tmp_mean_err = np.mean(err)
                # print(tmp_mean_err)
                for batch_idx, pred_success in enumerate(pred_successes):
                    if not pred_success and this_pred_steps[batch_idx] == self.T:
                        this_mj_pred_steps[batch_idx] = step
                if step == self.acc_err_steps:
                    this_mj_accumulated_pred_err = err

        pred_err_mean = np.mean(this_pred_err_hist)
        pred_err_std_mean = np.mean(this_pred_std_hist)
        pred_steps_mean = np.mean(this_pred_steps)
        this_accumulated_pred_err_mean = np.mean(this_accumulated_pred_err)
        loss_pred_steps_mean = np.mean(this_loss_pred_steps)
        self.pred_err = pred_err_mean
        self.pred_err_std = pred_err_std_mean
        self.pred_steps = pred_steps_mean
        self.pred_accumulated_err = this_accumulated_pred_err_mean
        self.loss_pred_steps = loss_pred_steps_mean
        self.policy.loss_pred_reliable_fwd_steps = self.loss_pred_steps

        if self.test_mujoco_err:
            mj_pred_err_mean = np.mean(this_mj_pred_err_hist)
            mj_pred_err_std_mean = np.mean(this_mj_pred_std_hist)
            mj_pred_steps_mean = np.mean(this_mj_pred_steps)
            mj_accumulated_pred_err_mean = np.mean(this_mj_accumulated_pred_err)
            self.mj_pred_err = mj_pred_err_mean
            self.mj_pred_err_std = mj_pred_err_std_mean
            self.mj_pred_steps = mj_pred_steps_mean
            self.mj_pred_accumulated_err = mj_accumulated_pred_err_mean


    def init_surprise_plot(self):
        self.surprise_fig = plt.figure(figsize=(10, 4), dpi=70)
        self.surprise_fig_ax = self.surprise_fig.add_subplot(111)
        self.surprise_fig_ax.set_facecolor('white')
        self.surprise_fig_ax.grid(color='gray', linestyle='-', linewidth=1)
        self.surprise_fig_ax.spines['left'].set_color('black')
        self.surprise_fig_ax.spines['bottom'].set_color('black')
        self.surprise_fig_ax.set_xlim([0, self.T])

        # init with some X and Y data
        x = [0]
        y = [0]
        self.surprise_fig_li_loss, = self.surprise_fig_ax.plot(x, y,color=(0,0,0), label='surprise')
        self.surprise_fig_li_pred_loss, = self.surprise_fig_ax.plot(x, y, color=(1, 0, 0), label='predicted surprise')
        self.surprise_fig_ax.relim()
        self.surprise_fig_ax.autoscale_view(True, True, True)
        plt.pause(0.01)
        self.surprise_fig.canvas.draw()
        plt.pause(0.01)
        plt.show(block=False)
        legend = plt.legend(frameon=0, loc='upper left')
        frame = legend.get_frame()
        frame.set_color('white')
        frame.set_linewidth(0)
        plt.pause(0.01)

    def replay_experience(self):
        if self.visualize_replay is False:
            plt.switch_backend('agg')
        if self.surprise_fig is None:
            self.init_surprise_plot()

        current_epoch = int(np.round(self.policy.model_replay_buffer.ep_no / self.episodes_per_epoch)) - 1

        last_added_idxs = np.argwhere(self.policy.model_replay_buffer.ep_added > (self.policy.model_replay_buffer.ep_no - self.episodes_per_epoch))
        last_added_idxs = last_added_idxs.flatten()
        last_added_losses = np.take(self.policy.model_replay_buffer.buffers['loss'], last_added_idxs, axis=0)
        last_added_max_losses = np.max(last_added_losses, axis=1)
        replay_idx = last_added_idxs[np.argmax(last_added_max_losses)]

        highest_mem_val = np.max(last_added_max_losses)
        if len(self.top_exp_replay_values) == self.top_exp_replay_values.maxlen:
            mem_val_required = min(self.top_exp_replay_values)
        else:
            mem_val_required = 0

        if highest_mem_val < mem_val_required:
            print("highes mem_val is {}, but {} required to be interesting enough for replay".format(highest_mem_val, mem_val_required))
            self.top_exp_replay_values = deque(np.array(self.top_exp_replay_values) * 0.95, maxlen=self.top_exp_replay_values.maxlen)
            return

        if replay_idx in self.replayed_episodes:
            return

        self.replayed_episodes.append(replay_idx)
        self.top_exp_replay_values.append(highest_mem_val)

        ep_added = self.policy.model_replay_buffer.ep_added[replay_idx]
        mem_val = self.policy.model_replay_buffer.memory_value[replay_idx]

        print("Replaying experience {} with highest memory value {}, added in episode {}.".format(replay_idx, mem_val, ep_added))

        replay_video_fpath = os.path.join(self.logger.get_dir(), "v_{:.2f}_ep_{}_.mp4".format(highest_mem_val, current_epoch))

        buff_idxs = [replay_idx]
        env = self.envs[0].env

        for buff_idx in buff_idxs:
            if current_epoch > 0:
                print("ep {}".format(current_epoch))

            if self.record_replay:
                frames = []
                plots = []

            for step_no, mj_state in enumerate(self.policy.model_replay_buffer.mj_states[buff_idx][:-1]):
                env.sim.set_state(mj_state)
                env.sim.forward()
                surprise_hist = self.policy.model_replay_buffer.buffers['loss'][buff_idx][:step_no+1]
                pred_surprise_hist = self.policy.model_replay_buffer.buffers['loss_pred'][buff_idx][:step_no+1]
                steps = list(range(step_no+1))

                plt.pause(0.0001)
                try:
                    self.surprise_fig_li_loss.set_xdata(steps)
                    self.surprise_fig_li_pred_loss.set_xdata(steps)
                except Exception as e:
                    print("Something went wrong: {}".format(e))
                self.surprise_fig_li_loss.set_ydata(surprise_hist)
                self.surprise_fig_li_pred_loss.set_ydata(pred_surprise_hist)
                try:
                    self.surprise_fig_ax.relim()
                except Exception as e:
                    print("Something went wrong: {}".format(e))
                self.surprise_fig_ax.autoscale_view(True, True, True)

                step_no += 1

                try:
                    self.surprise_fig.canvas.draw()
                except Exception as e:
                    print("Something went wrong: {}".format(e))

                if self.visualize_replay:
                    env.render()

                if self.record_replay:
                    record_viewer = env._get_viewer('rgb_array')
                    frame = record_viewer._read_pixels_as_in_window()
                    fig = self.surprise_fig
                    frame_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    frame_plot = frame_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plots.append(frame_plot)
                    frames.append(frame)
            if self.record_replay and len(frames) > 0:
                video_writer = imageio.get_writer(replay_video_fpath, fps=10)
                for frame, plot in zip(frames, plots):
                    f_width = frame.shape[1]
                    f_height = frame.shape[0]
                    p_width = plot.shape[1]
                    p_height = plot.shape[0]
                    pad_values = ((0, f_height-p_height),(f_width-p_width,0), (0,0))
                    plot_mask = np.where(plot == 255, np.uint8(1), np.uint8(0))
                    padded_plot_mask = np.pad(plot_mask, pad_values, 'constant', constant_values=1)
                    large_plot = np.pad(plot, pad_values, 'constant', constant_values=0)
                    masked_large_plot = large_plot * padded_plot_mask
                    frame_with_plot = (frame * padded_plot_mask) + masked_large_plot
                    try:
                        video_writer.append_data(frame_with_plot)
                    except Exception as e:
                        print("FS: {}, {}".format(frame.shape, frames[0].shape))
                        print("PS: {}, {}".format(plot.shape, plots[0].shape))
                        print(e)
                video_writer.close()








