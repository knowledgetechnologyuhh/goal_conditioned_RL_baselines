import numpy as np
import time, datetime

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from collections import deque
import numpy as np
import pickle
import copy
from mujoco_py import MujocoException
from baselines.her_pddl.pddl.pddl_util import obs_to_preds, gen_pddl_domain_problem, gen_plans
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


    def generate_rollouts(self, return_states=False):
        '''
        Overwrite generate_rollouts function from Rollout class with hierarchical rollout function that supports subgoals.
        :param return_states:
        :return:
        '''
        return self.generate_rollouts_hierarchical(return_states=return_states)

    def generate_rollouts_hierarchical(self, return_states=False):
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
        for t in range(self.T):
            if return_states:
                for i in range(self.rollout_batch_size):
                    mj_states[i].append(self.envs[i].env.sim.get_state())
            preds, n_hots = obs_to_preds(o, self.g, n_objects=self.n_objects)
            # TODO: For performance, perform planning only if preds has changed. May in addition use a caching approach where plans for known preds are stored.
            plans = gen_plans(preds, self.gripper_has_target, self.tower_height)
            self.subg = self.plans2subgoal(plans, o, self.g)
            for i, env in enumerate(self.envs):
                # print(env)
                env.env.goal = self.subg[i]
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
            subgoal_success = np.zeros(self.rollout_batch_size)
            overall_success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        subgoal_success[i] = info['is_success']
                        overall_success[i] = subgoal_success[i] and self.subg == self.g
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()


                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())

            successes.append(overall_success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            subgoals.append(self.subg.copy())
            o[...] = o_new
            ag[...] = ag_new
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

    def plans2subgoal(self, plans, obs, goals):
        subgoals = np.zeros(goals.shape)
        for i, (p,o,g) in enumerate(zip(plans, obs, goals)):
            # if len(p[0]) == 0:
            #     print("Empty plan now: {}".format(datetime.datetime.now()))
            subgoal = self.plan2subgoal(p,o,g)
            subgoals[i] = subgoal
        return subgoals

    def plan2subgoal(self, plan, obs, goal):
        # This currently only works for the environment TowerBuildMujocoEnv-sparse-gripper_random-o1-h1-1-v1. TODO: Make more general.
        if self.env_name != 'TowerBuildMujocoEnv-sparse-gripper_random-o1-h1-1':
            print("Subgoals currently only work for env TowerBuildMujocoEnv-sparse-gripper_random-o1-h1-1")
            return goal

        def get_o_pos(obs, o_idx):
            start_idx = (o_idx + 1) * 3
            end_idx = start_idx + 3
            o_pos = obs[start_idx:end_idx]
            return o_pos
        subgoal = copy.deepcopy(goal)
        actions_to_skip = ['open_gripper', 'grasp__o0'] # If we want to make use from these actions as well, the gripper opening value must be involved in the goal.
        for action in plan[0]:
            if action in actions_to_skip:
                continue
            o0_pos = get_o_pos(obs,0)
            if action == 'move_gripper_to__o0':
                # First three elements of goal represent target gripper pos.
                subgoal[:3] = o0_pos # Gripper should be above (at) object
                subgoal[3:] = o0_pos # Object should stay where it is
            elif action == 'move__o0_to_target':
                subgoal[:3] = subgoal[3:] # Gripper should be at object goal
            elif action == 'move_gripper_to_target':
                subgoal = subgoal # Gripper should be at gripper goal
            # print("Current subgoal action: {}".format(action))
            break # Stop after first useful action has been found.
        return subgoal


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

    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        for cyc in range(n_episodes):
            ro_start = time.time()
            episode = self.generate_rollouts()
            self.policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_train_batches):
                self.policy.train()
            self.policy.update_target_net()
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
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

        return logger(logs, prefix)

