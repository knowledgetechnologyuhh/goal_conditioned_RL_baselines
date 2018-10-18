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
        self.rollouts_per_epoch = 0


    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        for i,l in enumerate(self.avg_epoch_losses):
            logs += [('loss-{}'.format(i), l)]
        # if self.custom_histories:
        #     logs += [('mean_Q', np.mean(self.custom_histories[0]))]
        epoch_mean_pred_err = np.mean(self.err_history[-self.rollouts_per_epoch:])
        epoch_mean_pred_steps = np.mean(self.pred_history[-self.rollouts_per_epoch:])
        logs += [('pred_err', epoch_mean_pred_err)]
        logs += [('pred_steps', epoch_mean_pred_steps)]
        logs += [('episode', self.n_episodes)]

        return logger(logs, prefix)

    def generate_rollouts_update(self, n_cycles, n_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        self.avg_epoch_losses = []
        self.rollouts_per_epoch = n_cycles * n_batches
        for cyc in tqdm(range(n_cycles)):
            ro_start = time.time()
            episode = self.generate_rollouts()
            self.test_prediction_error(episode)
            self.policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_batches):
                losses = self.policy.train()
                if not isinstance(losses, tuple):
                    losses = [losses]
                self.avg_epoch_losses = [0 for _ in losses]
                for idx, loss in enumerate(losses):
                    self.avg_epoch_losses[idx] += loss
            dur_train += time.time() - train_start
        for idx,loss in enumerate(self.avg_epoch_losses):
            self.avg_epoch_losses[idx] = loss / n_batches / n_cycles
        self.draw_err_hist()
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)

        return updated_policy, time_durations

    def test_prediction_error(self, episode):
        ep_transitions = []
        for i1, eps_o in enumerate(episode['o']):
            transitions = []
            for i2,ep_o in enumerate(eps_o[:-1]):
                # for i3,o in enumerate(ep_o[:-1]):
                o = eps_o[i2]
                o2 = eps_o[i2 + 1]
                u = episode['u'][i1][i2]
                transitions.append({"o": o, "o2": o2, "u": u})

            # TO DO: In case of using a RNN as prediction model, reset states after each episode here.
            # Test error for each individual transition
            for t in transitions:
                o = t['o']
                u = t['u']
                o2 = t['o2']
                o2_pred = self.policy.forward_step(u,o)
                err = abs(o2 - o2_pred)
                norm_err = np.linalg.norm(o2 - o2_pred, axis=-1)
                self.err_history.append(norm_err)

            # TO DO: In case of using a RNN as prediction model, reset states after each episode here.
            # Test how many steps can be predicted without the error exceeding the goal achievement threshold.
            o = transitions[0]['o']
            step = 0
            for t in transitions[:-1]:
                step += 1
                u = t['u']
                o2 = t['o2']
                o = self.policy.forward_step(u, o)
                o_pred_g = self.policy.env._obs2goal(o)
                o_g = self.policy.env._obs2goal(o2)
                pred_success = self.policy.env._is_success(o_pred_g, o_g)
                if not pred_success:
                    break
            self.pred_history.append(step-1)


    def draw_err_hist(self):
        plt.clf()
        # plt.figure(figsize=(20, 8))
        fig = plt.figure(figsize=(10, 5))
        plt.semilogy(self.err_history)
        plt.savefig(self.err_hist_fname)

        plt.clf()
        # plt.figure(figsize=(20, 8))
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.pred_history)
        plt.savefig(self.pred_hist_fname)
        pass


