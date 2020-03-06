import numpy as np
from collections import deque
import time, sys
from baselines.template.util import convert_episode_to_batch_major, store_args
from baselines.template.rollout import Rollout
from tqdm import tqdm
from baselines.hac.utils import print_summary

class RolloutWorker(Rollout):


    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

        self.env = self.policy.env
        self.env.visualize = render
        self.FLAGS = self.policy.FLAGS
        self.T = T
        self.graph = kwargs['graph']

        if kwargs['print_summary']:
            print_summary(self.FLAGS, self.env)

        self.eval_data = {}

    def train_policy(self, n_train_rollouts, n_train_batches):
        successful_train_episodes = 0
        dur_train = 0
        dur_ro = 0
        for episode in tqdm(range(n_train_rollouts), file=sys.__stdout__, desc='Train Rollout'):
            ro_start = time.time()
            success, self.eval_data, train_duration = self.policy.train(self.env, episode, self.eval_data, n_train_batches)
            dur_train += train_duration

            if success:
                successful_train_episodes += 1

            self.n_episodes += 1
            dur_ro += time.time() - ro_start

        success_rate = 0
        if n_train_rollouts > 0:
            success_rate = successful_train_episodes / n_train_rollouts
        self.success_history.append(success_rate)
        return dur_train, dur_ro

    def generate_rollouts_update(self, n_train_rollouts, n_train_batches):
        dur_start = time.time()
        self.policy.FLAGS.test = False

        # TODO
        #  episode = self.generate_rollouts()
        #  self.policy.store_episode(episode)

        dur_train, dur_ro = self.train_policy(n_train_rollouts, n_train_batches)

        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)
        updated_policy = self.policy
        return updated_policy, time_durations

    def generate_rollouts(self, return_states=False):
        self.reset_all_rollouts()
        self.policy.FLAGS.test = True
        success, self.eval_data, _ = self.policy.train(self.env, self.n_episodes, self.eval_data, 0)
        self.success_history.append(1.0 if success else 0.0)
        self.n_episodes += 1

        return self.eval_data

    def current_mean_Q(self):
        return np.mean(self.custom_histories[0])

    def logs(self, prefix=''):
        eval_data = self.eval_data

        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('episodes', self.n_episodes)]

        for i in range(10):
            layer_prefix = '{}_{}/'.format(prefix, i)
            if "{}subgoal_succ".format(layer_prefix) in eval_data.keys():
                subg_succ_rate = eval_data["{}subgoal_succ".format(layer_prefix)] / eval_data["{}n_subgoals".format(layer_prefix)]
                eval_data['{}subgoal_succ_rate'.format(layer_prefix)] = subg_succ_rate

            if "{}Q".format(layer_prefix) in eval_data.keys():

                if "{}n_subgoals".format(layer_prefix) in eval_data.keys():
                    n_qvals = eval_data[
                            "{}n_subgoals".format(layer_prefix)]
                else:
                    n_qvals = 1

                avg_q = eval_data["{}Q".format(layer_prefix)] / n_qvals
                eval_data["{}avg_Q".format(layer_prefix)] = avg_q

        for k,v in sorted(eval_data.items()):
            if k.startswith(prefix):
                logs += [(k , v)]

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

