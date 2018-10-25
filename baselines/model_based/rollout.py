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

        self.err = None
        self.err_std = None
        self.pred = None
        self.loss_histories = []
        self.surprise_fig = None
        self.replayed_episodes = []
        self.top_exp_replay_values = deque(maxlen=10)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        for i,l in enumerate(self.loss_histories):
            if len(l) > 0:
                logs += [('loss-{}'.format(i), l[-1])]
            if len(l) > 1:
                loss_grad = l[-1] / l[-2]
            else:
                loss_grad = np.nan
            logs += [('loss-{}-grad'.format(i), loss_grad)]

        if self.err is not None :
            logs += [('pred_err', self.err)]
        if self.pred is not None:
            logs += [('pred_steps', self.pred)]
        logs += [('episode', self.n_episodes)]

        return logger(logs, prefix)

    def generate_rollouts_update(self, n_cycles, n_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        avg_epoch_losses = []
        # rollouts_per_epoch = n_cycles * self.rollout_batch_size
        last_episode_batch = None
        for cyc in range(n_cycles):
            print("episode {} / {}".format(cyc, n_cycles))
            ro_start = time.time()
            # episode, initial_mj_states = self.generate_rollouts(return_initial_states=True)
            episode, mj_states = self.generate_rollouts(return_states=True)
            last_episode_batch = episode
            # stored_idxs = self.policy.store_episode(episode, initial_mj_states=initial_mj_states)
            stored_idxs = self.policy.store_episode(episode, mj_states=mj_states)
            for i in stored_idxs:
                if i in self.replayed_episodes:
                    self.replayed_episodes.remove(i)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_batches):
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
            avg_loss = loss / n_cycles
            self.loss_histories[idx].append(avg_loss)
        self.test_prediction_error(last_episode_batch)
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        self.replay_experience()
        # plot_model_train("/".join(self.logger.get_dir().split("/")[:-1]))
        self.policy.model_replay_buffer.recompute_memory_values()
        return updated_policy, time_durations


    def test_prediction_error(self, episode):
        fwd_step = self.policy.forward_step_single
        this_err_hist = []
        this_std_hist = []
        this_pred_hist = []
        ep_transitions = []
        for i1, eps_o in enumerate(episode['o']):
            transitions = []
            for i2,ep_o in enumerate(eps_o[:-1]):
                # for i3,o in enumerate(ep_o[:-1]):
                o = eps_o[i2]
                o2 = eps_o[i2 + 1]
                u = episode['u'][i1][i2]
                transitions.append({"o": o, "o2": o2, "u": u})

            # Test error for each individual transition
            ep_err_hist = []
            s = None
            for t in transitions:
                o = t['o']
                u = t['u']
                o2 = t['o2']
                o2_pred, s = fwd_step(u,o,s)
                err = np.mean(abs(o2 - o2_pred))
                ep_err_hist.append(err)
            ep_err_mean = np.mean(ep_err_hist)
            ep_err_std = np.std(ep_err_hist)
            this_err_hist.append(ep_err_mean)
            this_std_hist.append(ep_err_std)

            # Test how many steps can be predicted without the error exceeding the goal achievement threshold.
            o = transitions[0]['o']
            step = 0
            s = None
            for t in transitions[:-1]:
                step += 1
                u = t['u']
                o2 = t['o2']
                o, s = fwd_step(u, o, s)
                o_pred_g = self.policy.env._obs2goal(o)
                o_g = self.policy.env._obs2goal(o2)
                pred_success = self.policy.env._is_success(o_pred_g, o_g)
                if not pred_success:
                    break
            this_pred_hist.append(step-1)


        err_mean = np.mean(this_err_hist)
        err_std_mean = np.mean(this_std_hist)
        pred_mean = np.mean(this_pred_hist)
        self.err = err_mean
        self.err_std = err_std_mean
        self.pred = pred_mean

    def init_surprise_plot(self):
        plt.clf()
        self.surprise_fig = plt.figure()
        self.surprise_fig_ax = self.surprise_fig.add_subplot(111)

        # some X and Y data
        x = [0]
        y = [0]
        self.surprise_fig_li, = self.surprise_fig_ax.plot(x, y)
        self.surprise_fig_ax.relim()
        self.surprise_fig_ax.autoscale_view(True, True, True)
        plt.ylabel('surprise (model-learning loss)')
        plt.pause(0.01)
        self.surprise_fig.canvas.draw()
        plt.pause(0.01)
        plt.show(block=False)
        plt.pause(0.01)

    # def start_record_experience_replay(self, video_path):
    #     if self._record_video:
    #         fps = (1 / self._time_per_render)
    #
    #         self._video_process = Process(target=self.save_video,
    #                                       args=(self._video_queue, video_path, fps))
    #         self._video_process.start()
    #
    #     if not self._record_video:
    #         self._video_queue.put(None)
    #         self._video_process.join()
    #         self._video_idx += 1
    #
    # def save_video(queue, filename, fps):
    #     writer = imageio.get_writer(filename, fps=fps)
    #     while True:
    #         frame = queue.get()
    #         if frame is None:
    #             break
    #         writer.append_data(frame)
    #     writer.close()

    def replay_experience(self):
        if self.surprise_fig is None:
            self.init_surprise_plot()

        replay_idx = np.argmax(self.policy.model_replay_buffer.memory_value)

        highest_mem_val = self.policy.model_replay_buffer.memory_value[replay_idx]
        if len(self.top_exp_replay_values) > 0:
            mem_val_required = min(self.top_exp_replay_values)
        else:
            mem_val_required = 0

        if highest_mem_val < mem_val_required:
            print("highes mem_val is {}, but {} required to be interesting enough for replay".format(highest_mem_val, mem_val_required))
            self.top_exp_replay_values *= 0.9
            return

        if replay_idx in self.replayed_episodes:
            return

        self.replayed_episodes.append(replay_idx)

        age = self.policy.model_replay_buffer.ep_added[replay_idx]
        mem_val = self.policy.model_replay_buffer.memory_value[replay_idx]
        init_max_surprise = max(self.policy.model_replay_buffer.loss_history[replay_idx])

        print("Replaying experience {} with highest memory value {}, age {}, initial max surprise {}".format(replay_idx, mem_val, age, init_max_surprise))

        buff_idxs = [replay_idx]
        env = self.envs[0].env

        for buff_idx in buff_idxs:

            step_no = 0
            for o,o2,u in zip(self.policy.model_replay_buffer.buffers['o'][buff_idx],
                              self.policy.model_replay_buffer.buffers['o2'][buff_idx],
                              self.policy.model_replay_buffer.buffers['u'][buff_idx]):
                next_o, _, _, _ = env.step(u)
                env.sim.set_state(self.policy.model_replay_buffer.mj_states[buff_idx][step_no])
                obs_err = np.mean(abs(next_o['observation'] - o2))
                env.render()

                # surprise = self.policy.model_replay_buffer.loss_history[buff_idx][step_no]
                surprise_hist = self.policy.model_replay_buffer.loss_history[buff_idx][:step_no+1]
                surprise = surprise_hist[-1]
                steps = list(range(step_no+1))

                plt.pause(0.0001)
                self.surprise_fig_li.set_xdata(steps)
                self.surprise_fig_li.set_ydata(surprise_hist)
                self.surprise_fig_ax.relim()
                self.surprise_fig_ax.autoscale_view(True, True, True)
                self.surprise_fig.canvas.draw()

                step_no += 1

                viewer = env._get_viewer()
                viewer.add_overlay(mj_const.GRID_TOPRIGHT, "Surprise:", "{:.2f}".format(surprise))
                viewer.add_overlay(mj_const.GRID_TOPRIGHT, "Observation error:", "{:.2f}".format(obs_err))

                #
                # y = step_no
                # z = step_no
                # dim1 = step_no % 100 * 1
                # dim2 = step_no % 100 * 1
                # dim3 = step_no % 3 * 1
                # img1 = np.ones([dim1, dim2, dim3], dtype=np.uint8) * 1
                # # img2 = np.ones([3, 100, 100], dtype=np.uint8) * step_no*10
                # # img3 = np.ones([100, 3, 100], dtype=np.uint8) * step_no*100
                # imgs = [img1]
                # for img in imgs:
                #     print(dim1,dim2,dim3, step_no)
                #     viewer.draw_pixels(img, y, z)
                #     # viewer.draw_pixels(img, "hallo", "TEst")
                #     # viewer.draw_pixels(y, z, img)
                #     # viewer.draw_pixels(y, img, z)






