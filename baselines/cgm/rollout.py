import time
import re
from collections import deque

import numpy as np
from mujoco_py import MujocoException

from baselines.util import convert_episode_to_batch_major
from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from baselines.cgm.cgm import CGM


class RolloutWorker(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        max_goal_len = 0
        if '-gripper_none-' not in kwargs['env_name']:
            max_goal_len += 3
        n_objects = re.search('-o[0-9]+-', kwargs['env_name']).group(0)
        n_objects = int(n_objects[2:-1])

        max_goal_len += (n_objects * 3)

        self.cgms = [CGM(max_goal_len) for _ in range(rollout_batch_size)]

        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        super().reset_rollout(i)
        self.cgms[i].sample_mask()


    def generate_rollouts_update(self, n_cycles, n_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        for cyc in range(n_cycles):
            # logger.info("Performing ")
            ro_start = time.time()
            episode = self.generate_rollouts()
            self.policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_batches):
                self.policy.train()
            self.policy.update_target_net()
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        return updated_policy, time_durations


    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # hold custom histories through out the iterations
        other_histories = []

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        for t in range(self.T):
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
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])

                    if 'is_success' in info:
                        success[i] = info['is_success']

                    # Here we ignore the successes received from the observation, and recompute them for CGM
                    self.cgms[i].compute_successes(self.envs[i].env._is_success,
                                                   curr_o_new['achieved_goal'], curr_o_new['desired_goal'])

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
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
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

        return convert_episode_to_batch_major(episode)

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




#
#
#
# from collections import deque
#
# import numpy as np
# import pickle
# import re
# from mujoco_py import MujocoException
# import os, time
#
# from baselines.util import convert_episode_to_batch_major, store_args
#
# class RolloutWorker:
#
#     @store_args
#     def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
#                  exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
#                  random_eps=0, history_len=100, render=False, **kwargs):
#         """Rollout worker generates experience by interacting with one or many environments.
#
#         Args:
#             make_env (function): a factory function that creates a new instance of the environment
#                 when called
#             policy (object): the policy that is used to act
#             dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
#             logger (object): the logger that is used by the rollout worker
#             rollout_batch_size (int): the number of parallel rollouts that should be used
#             exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
#                 current policy without any exploration
#             use_target_net (boolean): whether or not to use the target net for rollouts
#             compute_Q (boolean): whether or not to compute the Q values alongside the actions
#             noise_eps (float): scale of the additive Gaussian noise
#             random_eps (float): probability of selecting a completely random action
#             history_len (int): length of history for statistics smoothing
#             render (boolean): whether or not to render the rollouts
#         """
#         self.envs = [make_env() for _ in range(rollout_batch_size)]
#         assert self.T > 0
#
#         self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]
#
#         self.success_history = deque(maxlen=history_len)
#         self.Q_history = deque(maxlen=history_len)
#
#
#         max_goal_len = 0
#         if '-gripper_none-' not in kwargs['env_name']:
#             max_goal_len += 3
#         n_objects = re.search('-o[0-9]+-', kwargs['env_name']).group(0)
#         n_objects = int(n_objects[2:-1])
#
#         max_goal_len += (n_objects * 3)
#
#
#         self.goldilocks_sampling = kwargs['goldilocks_sampling']
#         # This causes the agent to disable goal masking during evaluation
#         if exploit is True:
#             self.goldilocks_sampling = 'none'
#         # if exploit:
#         #     self.goldilocks_sampling = 0.0
#         self.goal_mask_successes = {}
#         if self.goldilocks_sampling == 'none':
#             possible_goal_masks = ["".join(["1" for _ in range(max_goal_len)])]
#         else:
#             possible_goal_masks = []
#             max_n = pow(2, max_goal_len)
#             for n in range(max_n):
#                 mask_str = bin(n)[2:]
#                 mask_str = mask_str.rjust(max_goal_len, "0")
#                 possible_goal_masks.append(mask_str)
#         if 'stochastic3_' in self.goldilocks_sampling:
#             glr_avg_hist_len = int(self.goldilocks_sampling.split("_")[-1])
#         else:
#             glr_avg_hist_len = 10
#         for mask_str in possible_goal_masks:
#             self.goal_mask_successes[mask_str] = deque(maxlen=glr_avg_hist_len)
#
#         self.goal_slot_successes = ([deque(maxlen=glr_avg_hist_len) for _ in range(max_goal_len)], [deque(maxlen=glr_avg_hist_len) for _ in range(max_goal_len)])
#         self.subgoal_successes = [deque(maxlen=glr_avg_hist_len) for _ in range(max_goal_len)]
#
#
#         self.n_episodes = 0
#         self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
#         if 'gm' in dims.keys():
#             self.goal_mask = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goal mask
#         self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
#         self.initial_ag = np.empty((self.rollout_batch_size, self.dims['ag']), np.float32)  # achieved goals
#         self.reset_all_rollouts()
#         self.clear_history()
#
#         # do_write_mask_processes = False
#         self.do_write_mask_successes = ((not self.exploit))
#
#         self.write_mask_successes_file = os.path.join(logger.get_dir(), "mask_records.csv")
#         self.write_goal_slot_successes_file = os.path.join(logger.get_dir(), "slot_records.csv")
#         self.write_subgoal_successes_file = os.path.join(logger.get_dir(), "subgoal_records.csv")
#         if self.do_write_mask_successes:
#             # os.makedirs('mask_successes', exist_ok=True)
#             with open(self.write_mask_successes_file, 'w') as f:
#                 f.write("episode , time ")
#                 for m in sorted(self.goal_mask_successes.keys()):
#                     f.write(", {} (r) , {} (n)".format(m,m))
#                 f.write(", goal mask ")
#
#             with open(self.write_goal_slot_successes_file, 'w') as f:
#                 f.write("episode , time ")
#                 for m in range(max_goal_len):
#                     f.write(", {} (r0) , {} (n0) , {} (r1) , {} (n1)".format(m,m,m,m))
#
#         self.do_write_subgoal_successes = self.exploit and ('none' in self.goldilocks_sampling)
#         if self.do_write_subgoal_successes:
#             with open(self.write_subgoal_successes_file, 'w') as f:
#                 f.write("episode , time ")
#                 for m in range(max_goal_len):
#                     f.write(", {} (r0) , {} (n0) ".format(m,m))
#                 f.write(", goal mask ")
#
#
#     def reset_rollout(self, i):
#         """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
#         and `g` arrays accordingly.
#         """
#         # method_list = [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]
#         try:
#             self.envs[i].env.update_params({'gs_successes': self.goal_slot_successes})
#             if (self.exploit and ('none' in self.goldilocks_sampling)):
#                 if 'training_rollout_worker' in self.__dict__:
#                     for env in self.training_rollout_worker.envs:
#                         env.env.update_params({'subgoal_successes': self.subgoal_successes})
#                 self.envs[i].env.update_params({'subgoal_successes': self.subgoal_successes})
#             self.envs[i].env.update_params({'gm_successes': self.goal_mask_successes})
#             self.envs[i].env.update_params({'goldilocks_sampling': self.goldilocks_sampling})
#             self.envs[i].env.update_params({'mask_at_observation': self.mask_at_observation})
#         except:
#             # env does not support goal masking
#             pass
#
#         obs = self.envs[i].reset()
#         self.initial_o[i] = obs['observation']
#         self.initial_ag[i] = obs['achieved_goal']
#         self.g[i] = obs['desired_goal']
#         if 'goal_mask' in obs.keys():
#             self.goal_mask[i] = obs['goal_mask']
#
#
#     def reset_all_rollouts(self):
#         """Resets all `rollout_batch_size` rollout workers.
#         """
#         for i in range(self.rollout_batch_size):
#             self.reset_rollout(i)
#
#     def generate_rollouts(self):
#         """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
#         policy acting on it accordingly.
#         """
#         self.reset_all_rollouts()
#
#         # compute observations
#         o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
#         # hl_a = np.empty((self.rollout_batch_size, self.dims['hl_a']), np.float32)  # high_level actions
#         ag = np.empty((self.rollout_batch_size, self.dims['ag']), np.float32)  # achieved goals
#         o[:] = self.initial_o
#         # hl_a[:] = self.initial_hl_a
#         ag[:] = self.initial_ag
#
#         # generate episodes
#         obs, achieved_goals, acts, goals, goal_masks, successes = [], [], [], [], [], []
#         info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
#         Qs = []
#         for t in range(self.T):
#             if 'gm' in self.dims.keys():
#                 policy_output = self.policy.get_actions_mg(
#                     o, ag, self.g, self.goal_mask,
#                     compute_Q=self.compute_Q,
#                     noise_eps=self.noise_eps if not self.exploit else 0.,
#                     random_eps=self.random_eps if not self.exploit else 0.,
#                     use_target_net=self.use_target_net)
#             else:
#                 policy_output = self.policy.get_actions(
#                     o, ag, self.g,
#                     compute_Q=self.compute_Q,
#                     noise_eps=self.noise_eps if not self.exploit else 0.,
#                     random_eps=self.random_eps if not self.exploit else 0.,
#                     use_target_net=self.use_target_net)
#
#             if self.compute_Q:
#                 u, Q = policy_output
#                 Qs.append(Q)
#             else:
#                 u = policy_output
#
#             if u.ndim == 1:
#                 # The non-batched case should still have a reasonable shape.
#                 u = u.reshape(1, -1)
#
#             o_new = np.empty((self.rollout_batch_size, self.dims['o']))
#             # hl_a_new = np.empty((self.rollout_batch_size, self.dims['hl_a']))
#             ag_new = np.empty((self.rollout_batch_size, self.dims['ag']))
#             success = np.zeros(self.rollout_batch_size)
#             # compute new states and observations
#             for i in range(self.rollout_batch_size):
#                 try:
#                     # We fully ignore the reward here because it will have to be re-computed
#                     # for HER.
#                     curr_o_new, _, _, info = self.envs[i].step(u[i])
#                     goal_mask = curr_o_new['goal_mask']
#                     goal_mask_str = ''.join(map(str, goal_mask))
#                     if t == self.T - 1:
#                         if 'is_success' in info:
#                             success[i] = info['is_success']
#                             self.goal_mask_successes[goal_mask_str].append(info['is_success'])
#                             for gm_idx,gm_val in enumerate(goal_mask):
#                                 self.goal_slot_successes[gm_val][gm_idx].append(info['is_success'])
#                         if 'subgoal_successes' in info:
#                             [self.subgoal_successes[idx].append(info['subgoal_successes'][idx]) for idx in range(len(info['subgoal_successes'])) if goal_mask[idx] == 1]
#
#
#                         # if info['is_success'] ==1 :
#                         #     print("success")
#                     o_new[i] = curr_o_new['observation']
#                     # hl_a_new[i] = curr_o_new['hl_action']
#                     ag_new[i] = curr_o_new['achieved_goal']
#                     for idx, key in enumerate(self.info_keys):
#                         info_values[idx][t, i] = info[key]
#                     if self.render:
#                         self.envs[i].render()
#                 except MujocoException as e:
#                     return self.generate_rollouts()
#
#             if np.isnan(o_new).any():
#                 self.logger.warning('NaN caught during rollout generation. Trying again...')
#                 self.reset_all_rollouts()
#                 return self.generate_rollouts()
#
#             obs.append(o.copy())
#             achieved_goals.append(ag.copy())
#             successes.append(success.copy())
#             acts.append(u.copy())
#             goals.append(self.g.copy())
#             if 'gm' in self.dims.keys():
#                 goal_masks.append(self.goal_mask.copy())
#             o[...] = o_new
#             ag[...] = ag_new
#         obs.append(o.copy())
#         achieved_goals.append(ag.copy())
#         self.initial_o[:] = o
#
#         if 'gm' in self.dims.keys():
#             episode = dict(o=obs,
#                            u=acts,
#                            g=goals,
#                            ag=achieved_goals,
#                            gm=goal_masks)
#         else:
#             episode = dict(o=obs,
#                            u=acts,
#                            g=goals,
#                            ag=achieved_goals)
#         for key, value in zip(self.info_keys, info_values):
#             episode['info_{}'.format(key)] = value
#
#         # stats
#         successful = np.array(successes)[-1, :]
#         assert successful.shape == (self.rollout_batch_size,)
#         success_rate = np.mean(successful)
#         self.success_history.append(success_rate)
#         if self.compute_Q:
#             self.Q_history.append(np.mean(Qs))
#         self.n_episodes += self.rollout_batch_size
#
#         if self.do_write_mask_successes:
#             with open(self.write_mask_successes_file, 'a') as f:
#                 f.write("\n{} , {} ".format(self.n_episodes, time.strftime("%d.%m.%Y - %H:%M:%S")))
#                 for m in sorted(self.goal_mask_successes.keys()):
#                     succ_arr = self.goal_mask_successes[m]
#                     avg = np.nan_to_num(np.mean(succ_arr))
#                     n_tests = len(succ_arr)
#                     f.write(", {:.2f} , {}".format(avg,n_tests))
#                 f.write(" , {}".format(goal_mask_str))
#
#             with open(self.write_goal_slot_successes_file, 'a') as f:
#                 f.write("\n{} , {} ".format(self.n_episodes, time.strftime("%d.%m.%Y - %H:%M:%S")))
#                 for m in range(len(self.goal_slot_successes[0])):
#                     succ_arr_0 = self.goal_slot_successes[0][m]
#                     succ_arr_1 = self.goal_slot_successes[1][m]
#                     avg_0 = np.nan_to_num(np.mean(succ_arr_0))
#                     avg_1 = np.nan_to_num(np.mean(succ_arr_1))
#
#                     f.write(", {:.2f} , {} , {:.2f} , {}".format(avg_0, len(succ_arr_0), avg_1, len(succ_arr_1)))
#
#         if self.do_write_subgoal_successes:
#             with open(self.write_subgoal_successes_file, 'a') as f:
#                 f.write("\n{} , {} ".format(self.n_episodes, time.strftime("%d.%m.%Y - %H:%M:%S")))
#                 for m in range(len(self.subgoal_successes)):
#                     succ_arr = self.subgoal_successes[m]
#                     avg = np.nan_to_num(np.mean(succ_arr))
#                     f.write(", {:.2f} , {} ".format(avg, len(succ_arr)))
#                 f.write(" , {}".format(goal_mask_str))
#
#         return convert_episode_to_batch_major(episode)


