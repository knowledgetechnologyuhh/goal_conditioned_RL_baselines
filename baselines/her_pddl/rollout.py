import numpy as np
import time, datetime

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from collections import deque
import numpy as np
import pickle
import copy
from mujoco_py import MujocoException
from baselines.her_pddl.pddl.pddl_util import obs_to_preds, gen_pddl_domain_problem, gen_plans, plans2subgoals
from baselines.template.util import convert_episode_to_batch_major, store_args

class HierarchicalRollout(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, **kwargs)

        self.env_name = self.envs[0].env.spec._env_name
        self.n_objects = self.envs[0].env.n_objects
        self.gripper_has_target = self.envs[0].env.gripper_goal != 'gripper_none'
        self.tower_height = self.envs[0].env.goal_tower_height
        self.subg = self.g
        self.rep_correct_history = deque(maxlen=history_len)
        self.subgoal_succ_history = deque(maxlen=history_len)

    def generate_rollouts(self, return_states=False):
        '''
        Overwrite generate_rollouts function from Rollout class with hierarchical rollout function that supports subgoals.
        :param return_states:
        :return:
        '''
        return self.generate_rollouts_hierarchical(return_states=return_states)

    def generate_rollouts_hierarchical(self, return_states=False):
        # plan_ignore_actions = ['open_gripper']
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        if return_states:
            mj_states = [[] for _ in range(self.rollout_batch_size)]

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # hold custom histories through out the iterations
        other_histories = []

        # generate episodes
        obs, achieved_goals, acts, goals, subgoals, successes, subgoal_successes = [], [], [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key
                       in self.info_keys]

        preds, one_hots = obs_to_preds(o, self.g, n_objects=self.n_objects)
        plans = gen_plans(preds, self.gripper_has_target, self.tower_height)
        init_plan_lens = [len(plans[i][0]) for i in range(len(plans))]
        plan_lens = init_plan_lens.copy()
        # next_subg = plans2subgoals(plans, o, self.g.copy(), self.n_objects, actions_to_skip=plan_ignore_actions)
        self.subg = plans2subgoals(plans, o, self.g.copy(), self.n_objects)
        #
        avg_pred_correct = 0

        for t in range(self.T):
            if return_states:
                for i in range(self.rollout_batch_size):
                    mj_states[i].append(self.envs[i].env.sim.get_state())
            for i, env in enumerate(self.envs):
                # print(env)
                env.env.goal = self.subg[i]
                env.env.final_goal = self.g[i]
            if self.policy_action_params:
                policy_output = self.policy.get_actions(o, ag, self.subg, **self.policy_action_params)
            else:
                policy_output = self.policy.get_actions(o, ag, self.subg)

            if isinstance(policy_output, np.ndarray):
                u = policy_output  # get the actions from the policy output since actions should be the first element
            else:
                u = policy_output[0]
                other_histories.append(policy_output[1:])
            try:
                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
            except:
                self.logger.warn('Action "u" is not a Numpy array.')
            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))

            # success = np.zeros(self.rollout_batch_size)
            subgoal_success = np.zeros(self.rollout_batch_size)
            overall_success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                # We fully ignore the reward here because it will have to be re-computed
                # for HER.
                curr_o_new, _, _, info = self.envs[i].step(u[i])
                o_new[i] = curr_o_new['observation']
                ag_new[i] = curr_o_new['achieved_goal']
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[key]
                if self.render:
                    self.envs[i].render()
                subgoal_success[i] = self.envs[i].env._is_success(ag_new[i], self.subg[i])
                overall_success[i] = self.envs[i].env._is_success(ag_new[i], self.g[i])
            # TODO: For performance, perform planning only if preds has changed. May in addition use a caching approach where plans for known preds are stored.
            preds, n_hots = obs_to_preds(o_new, self.g, n_objects=self.n_objects)
            self.policy.obs_to_preds_memory.store_sample_batch(n_hots, o_new, self.g)
            n_hots_from_model = self.policy.predict_representation({'obs': o_new, 'goals': self.g})
            avg_pred_correct += np.mean([str(n_hots[i]) == str(n_hots_from_model[i]) for i in range(self.rollout_batch_size)])


            # Compute subgoal success
            for i in range(self.rollout_batch_size):
                subgoal_success[i] = self.envs[i].env._is_success(ag_new[i], self.subg[i])
                overall_success[i] = self.envs[i].env._is_success(ag_new[i], self.g[i])
            new_plans = gen_plans(preds, self.gripper_has_target, self.tower_height)

            # if going backwards, i.e., if plans are getting longer again, stay with the previous plans.
            for i, (newp,p) in enumerate(zip(new_plans, plans)):
                if len(newp[0]) > 0:
                    if p[0] == newp[0][1:]:
                        new_plans[i] = p
            plans = new_plans
            next_subg = plans2subgoals(new_plans, o, self.g.copy(), self.n_objects)
            for i in range(self.rollout_batch_size):
                self.envs[i].env.goal = next_subg[i]
                if subgoal_success[i] > 0 and plan_lens[i] > len(new_plans[i][0]):
                    plan_lens[i] = len(new_plans[i][0])
                    n_goals_achieved = init_plan_lens[i] - plan_lens[i]
                    # if n_goals_achieved > 0:
                    #     print("Achieved subgoal {} of {}".format(n_goals_achieved, init_plan_lens[i]))
                if self.render:
                    self.envs[i].render()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(overall_success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            subgoals.append(self.subg.copy())
            o[...] = o_new
            ag[...] = ag_new
            self.subg = next_subg
        avg_subgoal_succ = np.mean([ip - p for ip, p in zip(init_plan_lens, plan_lens)])
        avg_subgoals = np.mean(init_plan_lens)
        self.subgoal_succ_history.append(avg_subgoal_succ / avg_subgoals)
        avg_pred_correct /= self.T
        self.rep_correct_history.append(avg_pred_correct)
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        if return_states:
            for i in range(self.rollout_batch_size):
                mj_states[i].append(self.envs[i].env.sim.get_state())

        self.initial_o[:] = o
        episode = dict(o=obs,
                       u=acts,
                       # g=goals,
                       g=subgoals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if other_histories:
            for history_index in range(len(other_histories[0])):
                self.custom_histories.append(deque(maxlen=self.history_len))
                self.custom_histories[history_index].append([x[history_index] for x in other_histories])
        self.n_episodes += self.rollout_batch_size

        if return_states:
            ret = convert_episode_to_batch_major(episode), mj_states
        else:
            ret = convert_episode_to_batch_major(episode)
        return ret


class RolloutWorker(HierarchicalRollout):

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
        HierarchicalRollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)
        self.rep_loss_history = []

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        rep_ce_loss = 0
        for cyc in range(n_episodes):
            ro_start = time.time()
            episode = self.generate_rollouts()
            self.policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_train_batches):
                self.policy.train()
                rep_ce_loss += self.policy.train_representation()
            self.policy.update_target_net()
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        if n_episodes > 0:
            rep_ce_loss /= (n_train_batches * n_episodes)
        else:
            rep_ce_loss = np.nan
        self.rep_loss_history.append(rep_ce_loss)
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
        logs += [('episode', self.n_episodes)]
        if len(self.rep_loss_history) > 0:
            logs += [('rep_ce_loss', np.mean(self.rep_loss_history))]
        if len(self.rep_correct_history) > 0:
            logs += [('rep_correct', np.mean(self.rep_correct_history))]
        if len(self.subgoal_succ_history) > 0:
            logs += [('subgoal successes', np.mean(self.subgoal_succ_history))]

        return logger(logs, prefix)

