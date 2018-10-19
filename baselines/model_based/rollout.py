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
        self.avg_epoch_losses = []
        self.err_history = []
        logdir = logger.get_dir()
        self.err_hist_fname = os.path.join(logdir, "err_hist.png")
        self.pred_history = []
        self.pred_hist_fname = os.path.join(logdir, "pred_hist.png")
        self.loss_history = []
        self.loss_hist_fname = os.path.join(logdir, "loss_hist.png")
        self.n_rollouts_for_test_prediction_per_epoch = 0


    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        for i,l in enumerate(self.avg_epoch_losses):
            logs += [('loss-{}'.format(i), l)]
        # if self.custom_histories:
        #     logs += [('mean_Q', np.mean(self.custom_histories[0]))]
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
        self.avg_epoch_losses = []
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
                if self.avg_epoch_losses == []:
                    self.avg_epoch_losses = [0 for _ in losses]
                for idx, loss in enumerate(losses):
                    self.avg_epoch_losses[idx] += loss
            dur_train += time.time() - train_start
        for idx,loss in enumerate(self.avg_epoch_losses):
            self.avg_epoch_losses[idx] = loss / n_cycles
        self.loss_history.append(np.mean(self.avg_epoch_losses))
        self.test_prediction_error(last_episode_batch)
        self.draw_err_hist()
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)

        return updated_policy, time_durations

    def test_prediction_error(self, episode):
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
                o2_pred, s = self.policy.forward_step(u,o,s)
                err = abs(o2 - o2_pred)
                norm_err = np.linalg.norm(o2 - o2_pred, axis=-1)
                ep_err_hist.append(norm_err)
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
                o, s = self.policy.forward_step(u, o, s)
                o_pred_g = self.policy.env._obs2goal(o)
                o_g = self.policy.env._obs2goal(o2)
                pred_success = self.policy.env._is_success(o_pred_g, o_g)
                if not pred_success:
                    break
            this_pred_hist.append(step-1)


        err_mean = np.mean(this_err_hist)
        std_mean = np.mean(this_std_hist)
        pred_mean = np.mean(this_pred_hist)
        self.err_history.append([err_mean, std_mean])
        self.pred_history.append(pred_mean)


    def draw_err_hist(self):
        plt.clf()
        fig = plt.figure(figsize=(10, 5))
        err_hist_mean = np.swapaxes(self.err_history,0,1)[0]
        err_hist_std = np.swapaxes(self.err_history, 0, 1)[1]
        plt.semilogy(err_hist_mean)
        err_hist_low = err_hist_mean - err_hist_std
        err_hist_high = err_hist_mean + err_hist_std
        # plt.semilogy(err_hist_low, color='green', linestyle='dashed', alpha=0.25)
        # plt.semilogy(err_hist_high, color='green', linestyle='dashed', alpha=0.25)


        plt.savefig(self.err_hist_fname)

        plt.clf()
        # plt.figure(figsize=(20, 8))
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.pred_history)
        plt.savefig(self.pred_hist_fname)

        plt.clf()
        # plt.figure(figsize=(20, 8))
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.savefig(self.loss_hist_fname)
        pass


