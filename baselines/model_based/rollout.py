import numpy as np

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class RolloutWorker(Rollout):
    @store_args
    def __init__(self, make_env, policy, dims, logger, T, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.
        """
        Rollout.__init__(self, make_env, policy, dims, logger, T, **kwargs)

        logdir = logger.get_dir()
        self.err_history = []
        self.err_hist_fname = os.path.join(logdir, "err_hist.png")
        self.err_std_history = []
        self.err_hist_fname = os.path.join(logdir, "err_std_hist.png")
        self.pred_history = []
        self.pred_hist_fname = os.path.join(logdir, "pred_hist.png")
        self.loss_histories = []
        self.loss_hist_fname = os.path.join(logdir, "loss_hist-{}.png")
        self.n_rollouts_for_test_prediction_per_epoch = 0


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

        if len(self.err_history) > 0:
            epoch_mean_pred_err = self.err_history[-1]
            logs += [('pred_err', epoch_mean_pred_err)]
        if len(self.pred_history) > 0:
            epoch_mean_pred_steps = self.pred_history[-1]
            logs += [('pred_steps', epoch_mean_pred_steps)]
        logs += [('episode', self.n_episodes)]

        return logger(logs, prefix)

    def generate_rollouts_update(self, n_cycles, n_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        avg_epoch_losses = []
        rollouts_per_epoch = n_cycles * self.rollout_batch_size
        self.n_rollouts_for_test_prediction_per_epoch = min(rollouts_per_epoch, self.rollout_batch_size)
        last_episode_batch = None
        for cyc in tqdm(range(n_cycles)):
            ro_start = time.time()
            episode = self.generate_rollouts()
            last_episode_batch = episode
            self.policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_batches):
                losses = self.policy.train()
                if not isinstance(losses, tuple):
                    losses = [losses]

                if self.loss_histories == []:
                    self.loss_histories = [[] for _ in losses]
                if avg_epoch_losses == []:
                    avg_epoch_losses = [0 for _ in losses]
                for idx, loss in enumerate(losses):
                    avg_epoch_losses[idx] += loss
            dur_train += time.time() - train_start
        for idx,loss in enumerate(avg_epoch_losses):
            avg_loss = loss / n_cycles
            self.loss_histories[idx].append(avg_loss)
        self.test_prediction_error(last_episode_batch)
        # self.draw_err_hist()
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)

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
            # s_single = None
            for t in transitions:
                o = t['o']
                u = t['u']
                o2 = t['o2']
                o2_pred, s = fwd_step(u,o,s)
                # o2_pred_single, s_single = self.policy.forward_step_single(u,o,s_single)
                # err_single_vs_hist = np.linalg.norm(o2_pred_single - o2_pred, axis=-1)
                # print(err_single_vs_hist)
                norm_err = np.linalg.norm(o2 - o2_pred, axis=-1)
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
        self.err_history.append(err_mean)
        self.err_std_history.append(err_std_mean)
        self.pred_history.append(pred_mean)


    # def draw_err_hist(self):
    #     mean_last = 10
    #     plt.clf()
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.semilogy(self.err_history)
    #     plt.title('Mean of last {} epochs: {}'.format(mean_last, np.mean(self.err_history[-10:])))
    #     plt.savefig(self.err_hist_fname)
    #
    #     plt.clf()
    #     # plt.figure(figsize=(20, 8))
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.plot(self.pred_history)
    #     plt.title('Mean of last {} epochs: {}'.format(mean_last, np.mean(self.pred_history[-10:])))
    #     plt.savefig(self.pred_hist_fname)
    #
    #     plt.clf()
    #     # plt.figure(figsize=(20, 8))
    #     fig = plt.figure(figsize=(10, 5))
    #
    #     # plt.plot(self.loss_history)
    #     # plt.title('Mean of last {} epochs: {}'.format(mean_last, np.mean(self.loss_history[-10:])))
    #     plt.savefig(self.loss_hist_fname)
    #     pass


