import numpy as np
import time
from baselines.template.util import store_args
from baselines.template.util import logger as log_formater
from baselines.template.rollout import Rollout
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
        self.all_succ_history = deque(maxlen=history_len)
        self.success = np.zeros(self.rollout_batch_size)
        self.this_T = policy.T
        self.current_t = [0 for _ in range(self.rollout_batch_size)]
        self.current_episode = {}
        if self.is_leaf is False:
            self.child_rollout = RolloutWorker(make_env, policy.child_policy, dims, logger,
                                               rollout_batch_size=rollout_batch_size,
                                               render=render, **kwargs)
            make_env = self.make_env_from_child
            self.test_subgoal_perc = kwargs['test_subgoal_perc']
        self.tmp_env_ctr = 0
        Rollout.__init__(self, make_env, policy, dims, logger,
                         rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, **kwargs)
        self.env_name = self.envs[0].env.spec._env_name
        self.n_objects = self.envs[0].env.n_objects
        self.gripper_has_target = self.envs[0].env.gripper_goal != 'gripper_none'
        self.tower_height = self.envs[0].env.goal_tower_height

        # self.current_episodes = None
        self.n_train_batches = 0
        assert self.rollout_batch_size == 1, "For hierarchical rollouts, only rollout_batch_size=1 is allowed."


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

    def finished(self):
        return self.current_t[0] == self.T

    def generate_rollouts(self, return_states=False):
        '''
        Overwrite generate_rollouts function from Rollout class with hierarchical rollout function that supports subgoals.
        :param return_states:
        :return:
        '''
        if self.h_level == 0:
            self.reset_all_rollouts()
        for i, env in enumerate(self.envs):
            if self.is_leaf:
                self.envs[i].env.goal = self.g[i].copy()
            if self.h_level == 0:
                self.envs[i].env.final_goal = self.g[i].copy()
            self.envs[i].env.goal_hierarchy[self.h_level] = self.g[i].copy()

        # compute observations
        o = np.zeros((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.zeros((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals

        # generate episodes
        # obs, achieved_goals, acts, goals, successes, penalties = [], [], [], [], [], []


        for t in range(self.current_t[0], self.this_T):
            # At the first step add the current observation.
            if t == 0:
                for i in range(self.rollout_batch_size):
                    this_obs = self.envs[i].env._get_obs()
                    o[i] = this_obs['observation']
                    ag[i] = this_obs['achieved_goal']
                self.current_episode['obs'].append(o.copy())
                self.current_episode['achieved_goals'].append(ag.copy())

            # TODO: add binary parameter whether to use mean success rate of this policy or of child policy.
            self.policy_action_params['success_rate'] = self.get_mean_succ_rate()
            u, q = self.policy.get_actions(o, ag, self.g, **self.policy_action_params)

            o_new = np.zeros((self.rollout_batch_size, self.dims['o']))
            ag_new = np.zeros((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            penalty = np.zeros((self.rollout_batch_size, 1))
            self.q_history.append(np.mean(q))
            g = self.g
            # If the child rollout is already finished, zero all data.
            if self.is_leaf is False and self.child_rollout.finished():
                    # o = np.zeros_like(o)
                    # ag = np.zeros_like(ag)
                    u = np.zeros_like(u)
                    g = np.zeros_like(g)
            else:

                # Action execution
                if self.is_leaf is False:
                    if t == self.this_T-1:
                        u = self.g.copy()  # For last step use final goal
                    self.child_rollout.g = u
                    if not self.child_rollout.finished():
                        self.child_rollout.generate_rollouts()
                else: # In final layer execute physical action
                    for i in range(self.rollout_batch_size):
                        self.envs[i].step(u[i])

                for i in range(self.rollout_batch_size):
                    new_obs = self.envs[i].env._get_obs()
                    o_new[i] = new_obs['observation']
                    ag_new[i] = new_obs['achieved_goal']
                    this_success = self.envs[i].env._is_success(ag_new[i], self.g[i])
                    # info = {'is_success': this_success}
                    success[i] = this_success
                    # self.current_episode['info_is_success']
                    # for idx, key in enumerate(self.info_keys):
                    #     self.current_episode['info_values'][idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()

                if self.is_leaf is False:
                    # Add penalization depending on child subgoal success
                    child_success = self.child_rollout.success.copy()
                    for i in range(self.rollout_batch_size):
                        # TODO: For future work, compare this on-policy subgoal testing with off-policy.
                        if np.random.random_sample() < self.test_subgoal_perc:
                            penalty[i, 0] = True if np.isclose(child_success[i], 0.) else False

            self.current_episode['obs'].append(o_new.copy())
            self.current_episode['achieved_goals'].append(ag_new.copy())
            self.current_episode['successes'].append(success.copy())
            self.current_episode['info_is_success'].append(np.expand_dims(success.copy(), axis=-1))
            self.current_episode['penalties'].append(penalty.copy())
            self.current_episode['acts'].append(u.copy())
            self.current_episode['goals'].append(g.copy())
            # o[...] = o_new
            # ag[...] = ag_new

            self.current_t[0] = t + 1
            if success[0] == 1:
                break

        if self.is_leaf and np.mean(self.current_episode['penalties']) > 0:
            assert False, "For lowest layer, penalty should always be zero."

        episode = dict(o=self.current_episode['obs'],
                       u=self.current_episode['acts'],
                       g=self.current_episode['goals'],
                       ag=self.current_episode['achieved_goals'],
                       p=self.current_episode['penalties'],
                       info_is_success=self.current_episode['info_is_success'])
        # for key, value in zip(self.info_keys, self.current_episode['info_values']):
        #     episode['info_{}'.format(key)] = value

        # stats
        self.success = np.array(self.current_episode['successes'])[-1, :]
        assert self.success.shape == (self.rollout_batch_size,)
        success_rate = np.mean(self.success)
        # self.latest_success_rate = success_rate

        self.success_history.append(success_rate)
        self.all_succ_history.append(success_rate)
        ret = convert_episode_to_batch_major(episode)
        if self.finished():
            self.policy.store_episode(ret)
            self.n_episodes += self.rollout_batch_size
        return ret

    def generate_rollouts_update(self, n_episodes, n_train_batches, store_episode=True):
        # Make sure that envs of policy are those of the respective rollout worker. Important, because otherwise envs of evaluator and worker will be confused.
        self.n_train_batches = n_train_batches
        self.policy.set_envs(self.envs)
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        for cyc in tqdm(range(n_episodes), disable=self.h_level > 0):
            ro_start = time.time()
            episode = self.generate_rollouts()
            # if store_episode:
            #     self.policy.store_episode(episode)
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
        self.current_t[i] = 0
        for key in ['obs', 'achieved_goals', 'acts', 'goals', 'successes', 'penalties', 'info_is_success']:
            self.current_episode[key] = []
        # self.current_episode['info_values'] = [np.empty((self.this_T, self.rollout_batch_size,
        #                                                  self.dims['info_' + key]),
        #                                                 np.float32) for key in self.info_keys]
        if not self.is_leaf:
            self.child_rollout.reset_rollout(i)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('rollouts', self.n_episodes)]
        logs += [('steps', self.n_episodes * self.this_T)]
        if len(self.q_loss_history) > 0 and len(self.pi_loss_history) > 0:
            logs += [('q_loss', np.mean(self.q_loss_history))]
            logs += [('pi_loss', np.mean(self.pi_loss_history))]
        logs += [('mean_Q', np.mean(self.q_history))]
        logs = log_formater(logs, prefix+"_{}".format(self.h_level))

        if self.is_leaf is False:
            child_logs = self.child_rollout.logs(prefix=prefix)
            logs += child_logs

        return logs

    def get_mean_succ_rate(self, n_past_episodes=50):
        n_idx = min(n_past_episodes, len(self.all_succ_history))
        if n_idx == 0:
            return 0
        else:
            last_suc = list(self.all_succ_history)[-n_idx:]
            return np.mean(last_suc)

    # def get_succ_rate_avg_grad(self, n_past_episodes=50):
    #     n_idx = min(n_past_episodes, len(self.all_succ_history))
    #     if n_idx <= 2:
    #         return 0
    #     last_idxs = int(n_idx/2)
    #     last_suc = list(self.all_succ_history)[-last_idxs:]
    #     last_mean = np.mean(last_suc)
    #     # pre_las_idxs = last_idxs
    #     prev_suc = list(self.all_succ_history)[-n_idx:last_idxs]
    #     prev_mean = np.mean(prev_suc)
    #
    #     grad = last_mean / prev_mean
    #
    #     else:
    #
    #         return np.mean(last_suc)


    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.custom_histories.clear()
        self.q_history.clear()
        self.pi_loss_history.clear()
        self.q_loss_history.clear()
        self.rep_correct_history.clear()
        if self.is_leaf is False:
            self.child_rollout.clear_history()