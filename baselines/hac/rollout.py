import numpy as np
import time

from baselines.template.util import store_args, logger
from mujoco_py import MujocoException
from baselines.template.rollout import Rollout
from tqdm import tqdm
from baselines.hac.utils import print_summary
import sys

NUM_BATCH = 50
TEST_FREQ = 2

class RolloutWorker(Rollout):


    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)
        self.graph = kwargs['graph']

        self.agent = self.policy.agent
        self.env = self.policy.env
        self.env.visualize = render
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

        self.n_epochs = self.agent.FLAGS.n_epochs

        # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
        self.num_train_episodes = self.agent.FLAGS.n_train_rollouts
        self.num_test_episodes = self.agent.FLAGS.n_test_rollouts

        self.successful_train_episodes = 0
        self.successful_test_episodes = 0


    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_start = time.time()
        dur_train = 0
        dur_ro = 0

        for batch in range(self.n_epochs):
            print("\n--- TRAINING epoch {}---".format(batch))
            self.agent.FLAGS.test = False
            # Evaluate policy every TEST_FREQ batches if interleaving training and testing
            self.eval_data = {}

            for episode in tqdm(range(self.num_train_episodes)):
                ro_start = time.time()

                if self.agent.FLAGS.verbose:
                    print("\nBatch %d, Episode %d" % (batch, episode))

                # Train for an episode
                success, self.eval_data, train_duration = self.agent.train(self.env, episode, self.total_train_episodes, self.eval_data)
                dur_train += train_duration

                if success:
                    if self.agent.FLAGS.verbose:
                        print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                    # Increment successful episode counter if applicable
                    self.successful_train_episodes += 1

                self.total_train_episodes += 1
                self.total_train_steps += self.agent.steps_taken

            # Save agent
            self.agent.save_model(batch)
            self.eval_data['train/total_episodes'] = self.total_train_episodes
            self.eval_data['train/epoch_episodes'] = self.num_train_episodes

            if self.mix_train_test:
                break_condition, test_duration = self.test(batch, episode)

                if break_condition:
                    break

            dur_ro += time.time() - ro_start

        dur_total = time.time() - dur_start
        time_durations = (dur_total, dur_ro, dur_train)
        updated_policy = self.agent
        return updated_policy, time_durations

    def test(self, batch, episode):
        break_condition = False
        # Finish evaluating policy if tested prior batch
        print("\n--- TESTING epoch {}---".format(batch))
        self.agent.FLAGS.test = True
        for episode in tqdm(range(self.num_test_episodes)):
            # Train for an episode
            success, self.eval_data, test_duration = self.agent.train(self.env,
                    episode, self.total_train_episodes, self.eval_data)

            if success:
                if self.agent.FLAGS.verbose:
                    print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                # Increment successful episode counter if applicable
                self.successful_test_episodes += 1

            # if FLAGS.train_only or (mix_train_test and batch % TEST_FREQ != 0):
            self.total_test_episodes += 1
            self.total_test_steps += self.agent.steps_taken
        # Log performance
        success_rate = 0
        if self.num_test_episodes > 0:
            success_rate = self.successful_test_episodes / self.num_test_episodes

        if self.agent.FLAGS.verbose:
            print("\nTesting Success Rate %.2f%%" % success_rate)

        self.eval_data['test/total_episodes'] = self.total_test_episodes
        self.eval_data['test/epoch_episodes'] = self.num_test_episodes
        self.eval_data = self.agent.prepare_eval_data_for_log(self.eval_data)
        self.agent.log_performance(success_rate, self.eval_data,
                steps=self.total_train_steps, episode=self.total_train_episodes, batch=batch)
        print("\n--- END TESTING ---\n")
        early_stop_col = self.FLAGS.early_stop_data_column
        if early_stop_col in self.eval_data.keys():
            early_stop_val = self.eval_data[early_stop_col]
            if self.FLAGS.early_stop_threshold <= early_stop_val:
                break_condition = True
        else:
            print("Warning, early stop column not in keys")

        for k,v in self.eval_data.items():
            gap = max(1, 30 - len(k))
            gap_str = " " * gap
            print("{}: {} {:.2f}".format(k, gap_str, v))

        return break_condition, test_duration


    def generate_rollouts(self, return_states=False):
        ret = None
        return ret

    def current_mean_Q(self):
        return np.mean(self.custom_histories[0])

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.custom_histories:
            logs += [('mean_Q', np.mean(self.custom_histories[0]))]
        logs += [('episode', self.n_episodes)]

        return logger(logs, prefix)

