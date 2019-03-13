import numpy as np
import time

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from tqdm import tqdm

class RolloutWorker(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.is_leaf = True
        self.child_rollout = None
        self.h_level = 0
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

    # def generate_rollouts_update(self, n_episodes, n_train_batches):
    #     dur_ro = 0
    #     dur_train = 0
    #     dur_start = time.time()
    #     for cyc in tqdm(range(n_episodes)):
    #         # logger.info("Performing ")
    #         ro_start = time.time()
    #         episode = self.generate_rollouts()
    #         self.policy.store_episode(episode)
    #         dur_ro += time.time() - ro_start
    #         train_start = time.time()
    #         for _ in range(n_train_batches):
    #             self.policy.train()
    #         self.policy.update_target_net()
    #         dur_train += time.time() - train_start
    #     dur_total = time.time() - dur_start
    #     updated_policy = self.policy
    #     time_durations = (dur_total, dur_ro, dur_train)
    #     return updated_policy, time_durations

    def train_policy(self, n_train_batches):
        for _ in range(n_train_batches):
            self.policy.train()  # train actor-critic
        if n_train_batches > 0:
            self.policy.update_target_net()
            if not self.is_leaf:
                self.child_rollout.train_policy(n_train_batches)

    def generate_rollouts_update(self, n_episodes, n_train_batches):
        # Make sure that envs of policy are those of the respective rollout worker. Important, because otherwise envs of evaluator and worker will be confused.
        self.policy.set_envs(self.envs)
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        rep_ce_loss = 0
        for cyc in tqdm(range(n_episodes), disable=self.h_level > 0):
            ro_start = time.time()
            episode = self.generate_rollouts()
            self.policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            self.train_policy(n_train_batches)
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        # if n_episodes > 0 and n_train_batches > 0:
        #     rep_ce_loss /= (n_train_batches * n_episodes)
        # else:
        #     rep_ce_loss = np.nan
        # self.rep_loss_history.append(rep_ce_loss)
        return updated_policy, time_durations

    def current_mean_Q(self):
        return np.mean(self.custom_histories[0])

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.custom_histories:
            logs += [('mean_Q', np.mean(self.custom_histories[0]))]
        # logs += [('episode', self.n_episodes)]

        return logger(logs, prefix)

