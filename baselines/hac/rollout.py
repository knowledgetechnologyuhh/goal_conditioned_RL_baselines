import numpy as np
from collections import deque
import time
from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from tqdm import tqdm
from baselines.hac.utils import print_summary
import tensorflow as tf

class RolloutWorker(Rollout):


    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

        self.env = self.policy.env
        self.env.visualize = render
        self.policy.FLAGS.show = render
        self.FLAGS = self.policy.FLAGS

        print_summary(self.FLAGS, self.env)

        if not self.FLAGS.test and not self.FLAGS.train_only:
            self.mix_train_test = True
        else:
            self.mix_train_test = False

        self.total_train_episodes = 0
        self.total_train_steps = 0
        self.total_test_episodes = 0
        self.total_test_steps = 0

        self.num_train_episodes = self.FLAGS.n_train_rollouts
        self.num_test_episodes = self.FLAGS.n_test_rollouts

        self.success_history = deque(maxlen=history_len)

    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_start = time.time()
        dur_train = 0
        dur_ro = 0

        for batch in range(n_train_batches):

            print("\n--- TRAINING epoch {}---".format(batch))

            self.successful_train_episodes = 0
            self.successful_test_episodes = 0
            self.policy.FLAGS.test = False
            self.eval_data = {}

            for episode in tqdm(range(n_episodes)):
                ro_start = time.time()

                if self.policy.FLAGS.verbose:
                    print("\nBatch %d, Episode %d" % (batch, episode))

                # Train for an episode
                train_start = time.time()
                success, self.eval_data, = self.policy.train(self.env, episode, self.total_train_episodes, self.eval_data)
                dur_train += time.time() - train_start

                if success:
                    if self.policy.FLAGS.verbose:
                        print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                    # Increment successful episode counter if applicable
                    self.successful_train_episodes += 1

                self.total_train_episodes += 1
                self.total_train_steps += self.policy.steps_taken

            # Save agent
            self.policy.save_model(batch)
            self.eval_data['train/total_episodes'] = self.total_train_episodes
            self.eval_data['train/epoch_episodes'] = self.num_train_episodes

            if self.mix_train_test:
                test_time = time.time()
                break_condition = self.test(batch, n_episodes)
                dur_ro += time.time() - test_time

                if break_condition:
                    break

            dur_ro += time.time() - ro_start

        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)

        # TODO
        self.policy.eval_data = self.eval_data
        updated_policy = self.policy
        return updated_policy, time_durations

    def test(self, batch, n_episodes):
        print("\n--- TESTING epoch {}---".format(batch))
        # Finish evaluating policy if tested prior batch

        break_condition = False
        self.policy.FLAGS.test = True

        for episode in tqdm(range(max(1, n_episodes // 3))):
            # Train for an episode
            success, self.eval_data = self.policy.train(self.env,
                    episode, self.total_train_episodes, self.eval_data)

            if success:
                if self.policy.FLAGS.verbose:
                    print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                # Increment successful episode counter if applicable
                self.successful_test_episodes += 1

            # if FLAGS.train_only or (mix_train_test and batch % TEST_FREQ != 0):
            self.total_test_episodes += 1
            self.total_test_steps += self.policy.steps_taken
        # Log performance
        success_rate = 0
        if self.num_test_episodes > 0:
            success_rate = self.successful_test_episodes / self.num_test_episodes

        if self.policy.FLAGS.verbose:
            print("\nTesting Success Rate %.2f%%" % success_rate)

        self.success_history.append(success_rate)
        self.eval_data['test/total_episodes'] = self.total_test_episodes
        self.eval_data['test/epoch_episodes'] = self.num_test_episodes
        self.eval_data = self.policy.prepare_eval_data_for_log(self.eval_data)
        self.policy.log_performance(success_rate, self.eval_data, steps=self.total_train_steps, episode=self.total_train_episodes, batch=batch)

        print("\n--- END TESTING ---\n")

        early_stop_col = self.FLAGS.early_stop_data_column
        if early_stop_col in self.eval_data.keys():
            early_stop_val = self.eval_data[early_stop_col]
            if self.FLAGS.early_stop_threshold <= early_stop_val:
                break_condition = True
        else:
            print("Warning, early stop column not in keys")

        #  for k,v in self.eval_data.items():
        #      gap = max(1, 30 - len(k))
        #      gap_str = " " * gap
        #      print("{}: {} {:.2f}".format(k, gap_str, v))

        return break_condition


    def generate_rollouts(self, return_states=False):
        ret = None
        return ret

    def current_mean_Q(self):
        return np.mean(self.custom_histories[0])

    def logs(self, prefix=''):
        eval_data = self.policy.eval_data
        logs = []

        for k,v in sorted(eval_data.items()):
            logs += [(k , v)]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
