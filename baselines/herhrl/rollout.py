import numpy as np
import time
from baselines.template.util import store_args
from baselines.template.util import logger as log_formater
from baselines.template.rollout import Rollout
from tqdm import tqdm
from collections import deque
from baselines.template.util import convert_episode_to_batch_major
import sys
from PIL import Image

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
            graph (boolean): whether or not to create the graph
        """
        self.current_logs = []
        self.exploit = exploit
        self.is_leaf = policy.child_policy is None
        self.h_level = policy.h_level
        dims = policy.input_dims
        self.rep_correct_history = deque(maxlen=history_len)
        self.q_loss_history = deque(maxlen=history_len)
        self.pi_loss_history = deque(maxlen=history_len)
        self.preproc_loss_history = deque(maxlen=history_len)
        self.q_history = deque(maxlen=history_len)
        self.subgoals_achieved_history = deque(maxlen=history_len)
        self.subgoals_given_history = deque(maxlen=history_len)
        self.success = 0
        self.this_T = policy.T
        self.current_t = 0
        self.current_episode = {}
        self.subgoals_achieved = 0
        self.final_goal_achieved = False
        self.subgoals_given = []
        self.render_mode = 'human'
        self.graph = kwargs['graph']

        self.total_steps = 0
        if self.is_leaf is False:
            self.child_rollout = RolloutWorker(make_env, policy.child_policy, dims, logger,
                                               rollout_batch_size=rollout_batch_size,
                                               render=render, exploit=exploit, **kwargs)
            make_env = self.make_env_from_child
            self.test_subgoal_perc = kwargs['test_subgoal_perc']
        self.tmp_env_ctr = 0
        Rollout.__init__(self, make_env, policy, dims, logger,
                         rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, T=self.this_T, **kwargs)
        self.env_name = self.first_env.env.spec._env_name

        # Set Noise coefficient for environments
        self.obs_noise_coefficient = kwargs['obs_noise_coeff']
        self.first_env.env.obs_noise_coefficient = self.obs_noise_coefficient

        self.n_train_batches = 0
        # assert self.rollout_batch_size == 1, "For hierarchical rollouts, only rollout_batch_size=1 is allowed."

    def make_env_from_child(self):
        env = self.child_rollout.envs[self.tmp_env_ctr]
        self.tmp_env_ctr += 1
        return env

    def train_policy(self, n_train_batches):
        q_losses, pi_losses, preproc_losses = [], [], []
        for _ in range(n_train_batches):
            # q_loss, pi_loss, preproc_loss = self.policy.train()  # train actor-critic
            losses = self.policy.train()
            q_losses.append(losses[0])
            pi_losses.append(losses[1])
            if len(losses) > 2:
                preproc_losses.append(losses[2])
        if n_train_batches > 0:
            self.policy.update_target_net()
            self.q_loss_history.append(np.mean(q_losses))
            self.pi_loss_history.append(np.mean(pi_losses))
            if len(preproc_losses) > 0:
                self.preproc_loss_history.append(np.mean(preproc_losses))
            if not self.is_leaf:
                self.child_rollout.train_policy(n_train_batches)

    def finished(self):
        if self.is_leaf:
            return self.current_t == self.this_T
        else:
            return self.current_t == self.this_T and self.child_rollout.finished()

    def generate_rollouts(self, return_states=False):
        '''
        Overwrite generate_rollouts function from Rollout class with hierarchical rollout function that supports subgoals.
        :param return_states:
        :return:
        '''
        if self.h_level == 0:
            self.reset_all_rollouts()   # self.g is set here
            # self.subgoals_given[0].append(self.g.copy())
            if self.render:
                self.first_env.render(mode=self.render_mode)
        if self.is_leaf:
            self.first_env.env.goal = self.g.copy()
        if self.h_level == 0:
            self.first_env.env.final_goal = self.g.copy()
        self.first_env.env.goal_hierarchy[self.h_level] = self.g.copy()

        # compute observations
        o = np.zeros((self.dims['o']), np.float32)  # observations
        ag = np.zeros((self.dims['g']), np.float32)  # achieved goals
        # last_subgoals_achieved = self.subgoals_achieved[0]
        # for t in range(self.current_t[0], self.this_T):
        for t in range(self.this_T):
            # print(t)
            self.total_steps += self.rollout_batch_size
            # At the first step add the current observation.
            if t == 0:
                this_obs = self.first_env.env._get_obs()
                o = this_obs['observation']
                ag = this_obs['achieved_goal']
                self.current_episode['obs'].append(np.expand_dims(o.copy(), axis=0))
                self.current_episode['achieved_goals'].append(np.expand_dims(ag.copy(), axis=0))

            self.policy_action_params['success_rate'] = self.get_mean_succ_rate()
            u, q = self.policy.get_actions(o, ag, self.g, **self.policy_action_params)
            scaled_u = self.policy.scale_and_offset_action(u)
            if self.graph:
                reset = t==0
                if self.h_level == 0:
                    self.first_env.env.add_graph_values("q-high", q, t, reset=reset)
                else:
                    self.first_env.env.add_graph_values("q-value", q, t, reset=reset)

            o_new = np.zeros((self.dims['o']))
            ag_new = np.zeros((self.dims['g']))
            success = 0
            penalty = 0
            self.q_history.append(np.mean(q))
            g = self.g

            # Action execution
            if self.is_leaf is False:
                # if t == self.this_T-1:
                #     scaled_u = self.g.copy()  # For last step use final goal
                #     u = self.policy.inverse_scale_and_offset_action(scaled_u)
                self.child_rollout.g = scaled_u.copy()
                self.subgoals_given.append(scaled_u.copy())

                if self.child_rollout.finished():
                    self.child_rollout.current_t = 0
                self.child_rollout.generate_rollouts()
            else: # In final layer execute physical action
                self.first_env.step(scaled_u)

            # check success condition and rendering
            obs_dict = self.first_env.env._get_obs()
            non_noisy_ag = self.first_env.env._obs2goal(obs_dict['non_noisy_obs'])
            o_new = obs_dict['observation']
            ag_new = obs_dict['achieved_goal']
            # On the top level, i.e., for the final goal, assess success by objective non-noisy comparison.
            # On other levels, success is subjective.
            if self.h_level == 0:
                this_success = self.first_env.env._is_success(non_noisy_ag, self.g)
            else:
                this_success = self.first_env.env._is_success(ag_new, self.g)

            success = this_success
            if self.render:
                self.first_env.render(mode=self.render_mode)

                # if t==0:
                #     im = Image.fromarray(img).resize(size=[480, 295])
                #     im.save("your_file.jpeg")
                # self.first_env.render(mode='rgb_array')

            if self.is_leaf is False:
                # Add penalization depending on child subgoal success
                child_success = np.isclose(self.child_rollout.success, 1.)
                if child_success:
                    self.subgoals_achieved += 1
                    penalty = False
                else:
                    penalty = True

            o = o_new
            ag = ag_new

            self.current_episode['obs'].append(np.expand_dims(o_new.copy(), axis=0))
            self.current_episode['achieved_goals'].append(np.expand_dims(ag_new.copy(), axis=0))
            self.current_episode['info_is_success'].append(np.expand_dims([success.copy()], axis=-1))
            self.current_episode['penalties'].append(np.expand_dims([penalty], axis=0))
            self.current_episode['acts'].append(np.expand_dims(u.copy(), axis=0))
            self.current_episode['goals'].append(np.expand_dims(g.copy(), axis=0))

            self.current_t = t + 1
            if int(success):
                if self.h_level == 0:
                    self.set_final_goal_achieved()
                break
            if self.finished():
                break

        if self.is_leaf and np.mean(self.current_episode['penalties']) > 0:
            assert False, "For lowest layer, penalty should always be zero."

        # Distinguish here between leaf and h_level==0? Should this not always mean finalize_episode?
        # 1 time finalize episode in high level should include 'action_steps' times finalize episode in lower level
        if self.is_leaf:
            self.finalize_episode()
            for key in ['obs', 'achieved_goals', 'acts', 'goals', 'successes', 'penalties', 'info_is_success']: #TODO What is this current_episode stuff for?
                self.current_episode[key] = []

        if self.h_level == 0:
            if self.current_t == self.this_T or self.final_goal_achieved:
                self.finalize_episode()

    def finalize_episode(self):
        episode = dict(o=self.current_episode['obs'],
                       u=self.current_episode['acts'],
                       g=self.current_episode['goals'],
                       ag=self.current_episode['achieved_goals'],
                       p=self.current_episode['penalties'],
                       info_is_success=self.current_episode['info_is_success'],
                       steps=list(np.ones_like(self.current_episode['info_is_success']) * len(self.current_episode['acts']))
                       )
        self.success = np.array(self.current_episode['info_is_success'])[-1][0]
        episode = self.zero_pad_episode(episode)
        ret = convert_episode_to_batch_major(episode)
        self.success_history.append(self.success)
        if self.exploit == False:
            self.policy.store_episode(ret)
        self.n_episodes += self.rollout_batch_size
        self.subgoals_achieved_history.append(self.subgoals_achieved)
        self.subgoals_given_history.append(len(self.subgoals_given))

    def zero_pad_episode(self, episode):
        for key in episode.keys():
            max_t = self.this_T
            if key in ['o', 'ag']:
                max_t += 1
            assert len(episode[key]) > 0, "Empty episodes not allowed."
            for t in range(len(episode[key]), max_t):
                episode[key].append(np.zeros_like(episode[key][0]))
        return episode

    def latest_pad_episode(self, episode):
        for key in episode.keys():
            max_t = self.this_T
            if key in ['o', 'ag']:
                max_t += 1
            assert len(episode[key]) > 0, "Empty episodes not allowed."
            latest = len(episode[key])
            for t in range(len(episode[key]), max_t):
                episode[key].append(episode[key][latest-1])
        return episode

    def generate_rollouts_update(self, n_episodes, n_train_batches, store_episode=True):
        # Make sure that envs of policy are those of the respective rollout worker.
        # Important, because otherwise envs of evaluator and worker will be confused.
        self.n_train_batches = n_train_batches
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        for cyc in tqdm(range(n_episodes), disable=self.h_level > 0, file=sys.__stdout__):
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

    def init_rollout(self, obs):
        self.g = obs['desired_goal']
        if self.is_leaf == False:
            self.child_rollout.init_rollout(obs)

    def set_final_goal_achieved(self):
        self.final_goal_achieved = True
        if not self.is_leaf:
            self.child_rollout.set_final_goal_achieved()

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        self.policy.set_envs(self.envs)
        self.final_goal_achieved = False
        self.reset_rollout()

    def reset_rollout(self, i=0):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        if self.h_level == 0:
            obs = self.first_env.reset()
            self.init_rollout(obs)
        self.current_t = 0
        self.subgoals_achieved = 0
        self.subgoals_given = []
        for key in ['obs', 'achieved_goals', 'acts', 'goals', 'successes', 'penalties', 'info_is_success']:
            self.current_episode[key] = []
        if not self.is_leaf:
            self.child_rollout.reset_rollout()

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if not self.is_leaf:
            logs += [('subgoals_achieved', np.mean(self.subgoals_achieved_history))]
            logs += [('subgoals_given', np.mean(self.subgoals_given_history))]
        logs += [('rollouts', self.n_episodes)]
        logs += [('steps', self.total_steps)]
        if len(self.q_loss_history) > 0 and len(self.pi_loss_history) > 0:
            logs += [('q_loss', np.mean(self.q_loss_history))]
            logs += [('pi_loss', np.mean(self.pi_loss_history))]
        if len(self.preproc_loss_history) > 0:
            logs += [('preproc_loss', np.mean(self.preproc_loss_history))]
        logs += [('mean_Q', np.mean(self.q_history))]
        this_prefix = prefix
        if self.h_level > 0:
            this_prefix += "_{}".format(self.h_level)
        logs = log_formater(logs, this_prefix)

        if self.is_leaf is False:
            child_logs = self.child_rollout.logs(prefix=prefix)
            logs += child_logs
        # self.current_logs = logs
        return logs

    def get_mean_succ_rate(self, n_past_entries=10):
        n_idx = min(n_past_entries, len(self.success_history))
        if n_idx == 0:
            return 0
        else:
            last_suc = list(self.success_history)[-n_idx:]
            return np.mean(last_suc)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.custom_histories.clear()
        self.subgoals_achieved_history.clear()
        self.subgoals_given_history.clear()
        self.q_history.clear()
        self.pi_loss_history.clear()
        self.q_loss_history.clear()
        self.rep_correct_history.clear()
        if self.is_leaf is False:
            self.child_rollout.clear_history()