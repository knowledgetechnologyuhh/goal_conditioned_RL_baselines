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
        self.T = T
        max_goal_len = 0
        gripper_goal = None
        if '-gripper_none-' not in kwargs['env_name']:
            gripper_goal = 'gripper_none'
            max_goal_len += 3
        n_objects = re.search('-o[0-9]+-', kwargs['env_name']).group(0)
        n_objects = int(n_objects[2:-1])

        max_goal_len += (n_objects * 3)

        # TODO (fabawi): curriculum sampling to accepts params from training script (click_options)
        self.cgms = [CGM(max_goal_len, curriculum_sampling='stochastic3_0.5_5_50_5', gripper_goal=gripper_goal) for _ in range(rollout_batch_size)]

        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        super().reset_rollout(i)
        self.cgms[i].update_and_reset()
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
                    self.envs[i].goal = self.g[i] # TODO (fabawi): check if this is actually correct
                    curr_o_new, _, _, info = self.envs[i].step(u[i])

                    if t == self.T - 1:
                        last_timestep = True
                    else:
                        last_timestep = False

                    # Here we ignore the successes received from the observation, and recompute them for CGM
                    success[i], curr_o_new['achieved_goal'], curr_o_new['desired_goal'], curr_o_new['observation'] = \
                        self.cgms[i].compute_successes(self.envs[i].env._is_success,
                                                       curr_o_new['achieved_goal'], curr_o_new['desired_goal'],
                                                       curr_o_new['observation'], last_timestep)

                    self.g[i] = curr_o_new['desired_goal'] # TODO (fabawi): check if updating the goal is necessary
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

