import numpy as np
import time, datetime

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from baselines.her.rollout import RolloutWorker as HER_RolloutWorker
from collections import deque
import numpy as np
import pickle
import copy
from mujoco_py import MujocoException
from baselines.template.util import convert_episode_to_batch_major, store_args


class HierarchicalRollout(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, n_subgoals, rollout_batch_size=1,
                 exploit=False, history_len=100, render=False, **kwargs):

        parent_dims = dims.copy()
        parent_dims['u'] = parent_dims['g']
        parent_T = int(T/n_subgoals)
        child_T = n_subgoals    #T #int(T/parent_T)
        # parent_policy = copy.deepcopy(policy)
        parent_policy = policy[0]
        child_policy = policy[1]
        # parent_policy.scope = policy[0].scope + '_parent'
        # print("parent_policy scope {}".format(parent_policy.scope))
        # parent_policy.input_dims['u'] = parent_dims['u']
        # parent_policy.dimu = parent_dims['u']
        Rollout.__init__(self, make_env, parent_policy, parent_dims, logger, parent_T,
                         rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, **kwargs)

        self.env_name = self.envs[0].env.spec._env_name
        self.n_objects = self.envs[0].env.n_objects
        self.gripper_has_target = self.envs[0].env.gripper_goal != 'gripper_none'
        self.tower_height = self.envs[0].env.goal_tower_height
        self.subg = self.g
        self.rep_correct_history = deque(maxlen=history_len)
        self.subgoal_succ_history = deque(maxlen=history_len)

        # print("policy dimu {}".format(child_policy.dimu))
        # TODO: add condition to make this more modular
        self.child_rollout = SubRollout(make_env, child_policy, dims, logger, child_T, self.envs,
                                        rollout_batch_size=rollout_batch_size,
                                        history_len=history_len, render=render, **kwargs)

    def generate_rollouts(self, child_n_episodes = 10, return_states=False):
        '''
        Overwrite generate_rollouts function from Rollout class with hierarchical rollout function that supports subgoals.
        :param return_states:
        :return:
        '''
        return self.generate_rollouts_hierarchical(child_n_episodes, return_states=return_states)

    def generate_rollouts_hierarchical(self, child_n_episodes, return_states=False):
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
        acts_next = []

        # TODO: check
        subgoal_success = np.zeros(self.rollout_batch_size)
        overall_success = np.zeros(self.rollout_batch_size)

        for t in range(int(self.T)):
            if t == self.T - 1:
                print("t/T = {}/{}".format(t, int(self.T)))
            if return_states:
                for i in range(self.rollout_batch_size):
                    mj_states[i].append(self.envs[i].env.sim.get_state())

            # # TODO: check
            # for i, env in enumerate(self.envs):
            #     # print(env)
            #     env.env.goal = self.subg[i]
            #     env.env.final_goal = self.g[i]

            # TODO: if not testing: add_noise to get_action if random_sample()>0.2 otherwise get_random_action
            # TODO: if testing: get_action
            # if self.policy_action_params:
            #     policy_output = self.policy.get_actions(o, ag, self.subg, **self.policy_action_params)
            # else:
            #     policy_output = self.policy.get_actions(o, ag, self.subg)
            # print('o {}'.format(o))
            # print('g {}'.format(self.g))
            if self.policy_action_params:
                policy_output = self.policy.get_actions(o, ag, self.g, **self.policy_action_params)
            else:
                policy_output = self.policy.get_actions(o, ag, self.g)

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
            child_o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            child_ag_new = np.empty((self.rollout_batch_size, self.dims['g']))

            # # TODO: check
            # subgoal_success = np.zeros(self.rollout_batch_size)
            # overall_success = np.zeros(self.rollout_batch_size)
            # action(i) --> subgoal(i-1)
            self.subg = u
            self.child_rollout.inherited_values(o, ag, self.subg)

            # TODO: check
            for i, env in enumerate(self.envs):
                # print("env.env.goal         {}".format(env.env.goal))
                # print("env.env.final_goal   {}".format(env.env.final_goal))
                env.env.goal = self.subg[i]
                env.env.final_goal = self.g[i]
                # print("CHANGING GOAL=======================")
                # print("env.env.goal         {}".format(env.env.goal))
                # print("env.env.final_goal   {}".format(env.env.final_goal))

            # TODO: HAC proposes to stop this earlier if self.subg achieves
            # for cyc in range(child_n_episodes):
            for cyc in range(1):
                # print("child_episode = {}/{}".format(cyc, child_n_episodes))
                episode = self.child_rollout.generate_rollouts()    # policy.get_actions & execute action happen inside
                self.child_rollout.policy.store_episode(episode)    # transition t0 is stored in replay buffer

                # # TODO: check
                # subgoal_success = np.zeros(self.rollout_batch_size)
                # overall_success = np.zeros(self.rollout_batch_size)
                # compute new states and observations
                # preds, n_hots = [], []  # TODO: check
                for i in range(self.rollout_batch_size):
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.

                    # if len(u[i]) == 3:
                    #     u_com = np.append(u[i], 1)
                    # else:
                    #     u_com = np.append(u[i, 0:3], 1)
                    # curr_o_new, _, _, info = self.envs[i].step(u_com)  # TODO: commented b/c u size is 3 or 6
                    curr_o_new = self.envs[i].env._get_obs()
                    child_curr_o_new = self.child_rollout.envs[i].env._get_obs()
                    # is_success = self.envs[i].env._is_success(curr_o_new['achieved_goal'], curr_o_new['desired_goal'])
                    is_success = self.child_rollout.envs[i].env._is_success(child_curr_o_new['achieved_goal'],
                                                                            child_curr_o_new['desired_goal'])
                    info = {
                        'is_success': is_success
                    }
                    # print("parent action    {}".format(u[i]))
                    # print("parent self.subg     {}".format(self.subg[i]))
                    # print("parent self.g        {}".format(self.g[i]))
                    # print("parent desired_goal  {}".format(curr_o_new['desired_goal']))
                    # print("parent achieved_goal {}".format(curr_o_new['achieved_goal']))

                    child_o_new[i] = child_curr_o_new['observation']
                    child_ag_new[i] = child_curr_o_new['achieved_goal']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    #
                    # print("child desired_goal   {}".format(child_curr_o_new['desired_goal']))
                    # print("parent desired_goal  {}".format(curr_o_new['desired_goal']))
                    # print("child achieved_goal  {}".format(child_curr_o_new['achieved_goal']))
                    # print("parent achieved_goal {}".format(curr_o_new['achieved_goal']))

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                    # subgoal_success[i] = self.envs[i].env._is_success(ag_new[i], self.subg[i])
                    subgoal_success[i] = is_success
                    overall_success[i] = self.envs[i].env._is_success(ag_new[i], self.g[i])

                # print("child_o_new {}".format(child_o_new))
                # print("o_new       {}".format(o_new))
                # print("child o next {}".format(self.child_rollout.initial_o))
                # print('child_rollout.success_history {}'.format(self.child_rollout.success_history))
                if np.any(subgoal_success > np.zeros(self.rollout_batch_size)+0.5) or \
                        np.any(overall_success > np.zeros(self.rollout_batch_size)+0.5):
                    print("parent self.subg {}".format(self.subg[i]))
                    print("parent self.g    {}".format(self.g[i]))
                    print("parent desired_goal  {}".format(curr_o_new['desired_goal']))
                    print("parent achieved_goal {}".format(curr_o_new['achieved_goal']))
                    print("parent is success    {}".format(is_success))
                    print(subgoal_success)
                    break
                    #
                    # if 'is_success' in info:
                    #     subgoal_success[i] = info['is_success']

                    # preds.append(self.envs[i].env.get_preds()[0])   # TODO: check
                    # n_hots.append(self.envs[i].env.get_preds()[1])
            # TODO: For performance, perform planning only if preds has changed. May in addition use a caching approach
            #  where plans for known preds are stored.
            # n_hots[0] = np.array([0,0,1])
            # self.policy.obs2preds_buffer.store_sample_batch(n_hots, o_new, self.g)  # TODO: check
            # n_hots_from_model = self.policy.predict_representation({'obs': o_new, 'goals': self.g}) # TODO: check
            # n_hots = np.array(n_hots)
            # avg_pred_correct += np.mean([str(n_hots[i]) == str(n_hots_from_model[i])
            #                              for i in range(self.rollout_batch_size)])

            # Compute subgoal and goal success
            # for i in range(self.rollout_batch_size):
            #     subgoal_success[i] = self.envs[i].env._is_success(ag_new[i], self.subg[i])
            #     overall_success[i] = self.envs[i].env._is_success(ag_new[i], self.g[i])
            if np.any(subgoal_success > np.zeros(self.rollout_batch_size)+0.5) or \
                        np.any(overall_success > np.zeros(self.rollout_batch_size)+0.5):
                print("sucess: subgoal = {}, overall = {}".format(subgoal_success, overall_success))

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(overall_success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            subgoals.append(self.subg.copy())

            u_next = np.empty(u.shape, dtype=np.float32)
            for i in range(self.rollout_batch_size):
                # u_next[i] = self.envs[i].env._obs2goal(o_new[i])
                u_next[i] = self.child_rollout.envs[i].env._obs2goal(self.child_rollout.initial_o)

            # print("child o next {}".format(self.child_rollout.initial_o))
            # print("u_next       {}".format(u_next))
            acts_next.append(u_next.copy())

            # o[...] = o_new
            # ag[...] = ag_new
            # self.subg = next_subg   # TODO: check subg things
        # avg_subgoal_succ = np.mean([ip - p for ip, p in zip(init_plan_lens, plan_lens)])
        # avg_subgoals = np.mean(init_plan_lens)
        # self.subgoal_succ_history.append(avg_subgoal_succ / avg_subgoals)
        self.subgoal_succ_history.append(self.child_rollout.success_history)
        # avg_pred_correct /= self.T
        # self.rep_correct_history.append(avg_pred_correct)

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        if return_states:
            for i in range(self.rollout_batch_size):
                mj_states[i].append(self.envs[i].env.sim.get_state())

        self.initial_o[:] = o
        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       # g=subgoals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value
            # print("episode {}".format(episode))

        # print('episode {}'.format(episode))

        # print("acts      {}".format(acts))
        # print("acts_next {}".format(acts_next))
        episode_normal = dict(o=obs,
                              u=acts_next,
                              # u=acts,
                              g=goals,
                              # g=subgoals,
                              ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode_normal['info_{}'.format(key)] = value
        # ret_normal = None
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
            # ret_normal = convert_episode_to_batch_major(episode_normal), mj_states
        else:
            ret = convert_episode_to_batch_major(episode)
            # ret_normal = convert_episode_to_batch_major(episode_normal)

        penalty = False
        # TODO: add penalized transition if subgoal_success doesn't have any positive value
        if np.all(subgoal_success < np.ones(self.rollout_batch_size)-0.5):
            print("PENALIZE SUBGOAL")
            penalty = True
        # else:
        #     ret = None
            # self.policy.store_episode(episode, penalty=True)
        return ret, penalty
        # return ret_normal, penalty, ret


class SubRollout(Rollout):
    @store_args
    def __init__(self, make_env, policy, dims, logger, T, envs, rollout_batch_size=1,
                 exploit=False, history_len=100, render=False, **kwargs):
        self.inherited_g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, **kwargs)
        self.envs = envs
        # self.success = np.zeros(self.rollout_batch_size)

    def inherited_values(self, o, ag, g):
        """
        Setting obsevation, achieved goal and goal value of child rollout using the parent values
        :param o: observation
        :param ag: accessible goal
        :param g: goal
        :return:
        """
        self.o = o.copy()
        self.ag = ag.copy()
        self.inherited_g = g.copy()
        # self.o = o
        # self.ag = ag
        # self.inherited_g = g

    def reset_all_rollouts(self):
        return self.set_all_values()

    def set_all_values(self):
        self.initial_o = self.o
        self.initial_ag = self.ag
        self.g  = self.inherited_g

    # def get_result
    #     super().generate_rollouts()
    #     self.success = successes


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
            episode, penalty = self.generate_rollouts(n_episodes)
            self.policy.store_episode(episode, penalty=penalty)
            self.policy.store_episode(episode)
            # print("parent cyc/n_episodes {}/{}".format(cyc,n_episodes))
            # episode_normal, penalty, episode = self.generate_rollouts(n_episodes)
            # if penalty:
            #     self.policy.store_episode(episode, penalty=penalty)
            # self.policy.store_episode(episode_normal)

            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_train_batches):
                # print("train {}/{}".format(_,n_train_batches))
                self.policy.train()     # train actor-critic
                # rep_ce_loss += self.policy.train_representation()   # TODO: check
            self.policy.update_target_net()

            for _ in range(n_train_batches):
                self.child_rollout.policy.train()
            self.child_rollout.policy.update_target_net()
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
        # TODO: check subg things
        if len(self.subgoal_succ_history) > 0:
            logs += [('subgoal successes', np.mean(self.subgoal_succ_history))]

        return logger(logs, prefix)

