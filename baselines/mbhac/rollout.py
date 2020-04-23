import numpy as np
import time
import sys
from baselines.template.util import store_args
from baselines.template.rollout import Rollout
from tqdm import tqdm

class RolloutWorker(Rollout):


    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

        self.env = self.policy.env
        self.env.visualize = render
        self.env.graph = kwargs['graph']
        self.eval_data = {}

    def train_policy(self, n_train_rollouts, n_train_batches):
        dur_train = 0
        dur_ro = 0

        for episode in tqdm(range(n_train_rollouts), file=sys.__stdout__, desc='Train Rollout'):
            ro_start = time.time()
            success, self.eval_data, train_duration = self.policy.train(self.env, episode, self.eval_data, n_train_batches)
            dur_train += train_duration
            self.success_history.append(1.0 if success else 0.0)
            self.n_episodes += 1
            dur_ro += time.time() - ro_start - train_duration

        return dur_train, dur_ro

    def generate_rollouts_update(self, n_train_rollouts, n_train_batches):
        dur_start = time.time()
        self.policy.set_train_mode()
        dur_train, dur_ro = self.train_policy(n_train_rollouts, n_train_batches)
        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)
        updated_policy = self.policy
        return updated_policy, time_durations

    def generate_rollouts(self, return_states=False):
        self.reset_all_rollouts()
        self.policy.set_test_mode()
        success, self.eval_data, _ = self.policy.train(self.env, self.n_episodes, self.eval_data, 0)
        self.success_history.append(1.0 if success else 0.0)
        self.n_episodes += 1
        return self.eval_data

    def logs(self, prefix=''):
        eval_data = self.eval_data

        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('episodes', self.n_episodes)]

        # Get metrics for all layers of the hierarchy
        for i in range(self.policy.n_layers):
            layer_prefix = '{}_{}/'.format(prefix, i)

            subg_succ_prefix = '{}subgoal_succ'.format(layer_prefix)
            if subg_succ_prefix in eval_data.keys():
                if len(eval_data[subg_succ_prefix]) > 0:
                    logs += [(subg_succ_prefix + '_rate', np.mean(eval_data[subg_succ_prefix]))]
                else:
                    logs += [(subg_succ_prefix + '_rate', 0.0)]

            n_subg_prefix = "{}n_subgoals".format(layer_prefix)
            if n_subg_prefix in eval_data.keys():
                logs += [(n_subg_prefix, eval_data[n_subg_prefix])]

            curi_prefix = "{}curiosity".format(layer_prefix)
            if curi_prefix in eval_data.keys():
                logs += [(curi_prefix, np.nanmean(eval_data[curi_prefix]))]

            mb_loss_prefix = "{}mb_loss".format(layer_prefix)
            if mb_loss_prefix in eval_data.keys():
                logs += [(mb_loss_prefix, eval_data[mb_loss_prefix])]

            q_prefix = "{}Q".format(layer_prefix)
            if q_prefix in eval_data.keys():
                if len(eval_data[q_prefix]) > 0:
                    logs += [("{}avg_Q".format(layer_prefix), np.mean(eval_data[q_prefix]))]
                else:
                    logs += [("{}avg_Q".format(layer_prefix), 0.0)]

        if prefix != '' and not prefix.endswith('/'):
            new_logs = []
            for key, val in logs:
                if not key.startswith(prefix):
                    new_logs +=[((prefix + '/' + key, val))]
                else:
                    new_logs += [(key, val)]
            logs = new_logs

        return logs

    def clear_history(self):
        self.success_history.clear()
        self.custom_histories.clear()
        if hasattr(self, 'eval_data'):
            self.eval_data.clear()

