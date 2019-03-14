import numpy as np
import time

from baselines.template.util import store_args
from baselines.template.util import logger as log_formater
from baselines.template.rollout import Rollout
from baselines.herhrl.sub_rollout import RolloutWorker as SubRolloutWorker
from tqdm import tqdm
from collections import deque
from baselines.template.util import convert_episode_to_batch_major

class RolloutWorker(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, rollout_batch_size=1,
                 exploit=False, history_len=200, render=False, **kwargs):
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
        self.exploit = exploit
        self.is_leaf = policy.child_policy is None
        self.h_level = policy.h_level
        dims = policy.input_dims
        self.rep_correct_history = deque(maxlen=history_len)
        self.q_loss_history = deque(maxlen=history_len)
        self.pi_loss_history = deque(maxlen=history_len)
        self.q_history = deque(maxlen=history_len)
        if self.is_leaf is False:
            self.child_rollout = SubRolloutWorker(make_env, policy.child_policy, dims, logger,
                                               rollout_batch_size=rollout_batch_size,
                                               render=render, **kwargs)

        make_env = self.make_env_from_child
        self.tmp_env_ctr = 0
        Rollout.__init__(self, make_env, policy, dims, logger,
                         rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, **kwargs)

    def make_env_from_child(self):
        env = self.child_rollout.envs[self.tmp_env_ctr]
        self.tmp_env_ctr += 1
        return env

    def train_policy(self, n_train_batches):
        q_losses, pi_losses = [], []
        for _ in range(n_train_batches):
            q_loss, pi_loss = self.policy.train()  # train actor-critic
            q_losses.append(q_loss)
            pi_losses.append(pi_loss)
        if n_train_batches > 0:
            self.policy.update_target_net()
            self.q_loss_history.append(np.mean(q_losses))
            self.pi_loss_history.append(np.mean(pi_losses))
            if not self.is_leaf:
                self.child_rollout.train_policy(n_train_batches)

    def generate_rollouts(self, return_states=False):
        self.reset_all_rollouts()
        self.child_rollout.g = self.g.copy()
        _, _, child_episodes = self.child_rollout.generate_rollouts_update(n_episodes=1, n_train_batches=0,
                                                                           store_episode=(self.exploit == False),
                                                                           return_episodes=True)
        return None

    def generate_rollouts_update(self, n_episodes, n_train_batches, store_episode=True, return_episodes=False):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        for cyc in tqdm(range(n_episodes), disable=self.h_level > 0):
            ro_start = time.time()
            self.generate_rollouts()
            dur_ro += time.time() - ro_start
            train_start = time.time()
            self.train_policy(n_train_batches)
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        ret = updated_policy, time_durations
        return ret

    def init_rollout(self, obs, i):
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']
        if self.is_leaf == False:
            self.child_rollout.init_rollout(obs, i)

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        if self.h_level == 0:
            obs = self.envs[i].reset()
            self.init_rollout(obs, i)



    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs = log_formater(logs, prefix+"_{}".format(self.h_level))
        if self.is_leaf is False:
            child_logs = self.child_rollout.logs(prefix=prefix)
            logs += child_logs
        return logs

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        if self.is_leaf is False:
            self.child_rollout.clear_history()