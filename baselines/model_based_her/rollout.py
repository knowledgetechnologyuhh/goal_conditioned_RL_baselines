import numpy as np
import time

from baselines.template.util import store_args, logger
from baselines.template.rollout import Rollout
from baselines.util import convert_episode_to_batch_major
from tqdm import tqdm
from collections import deque
import sys
from mujoco_py import MujocoException

class RolloutWorker(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, history_len=100, render=False, **kwargs):
        self.render_mode = 'human'
        self.graph = kwargs['graph']

        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)

        self.env_name = self.first_env.env.spec._env_name

        self.replayed_trajectories = []
        self.loss_histories = []

    def generate_rollouts(self, return_states=False):
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
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        for t in range(self.T):
            if return_states:
                for i in range(self.rollout_batch_size):
                    mj_states[i].append(self.envs[i].env.sim.get_state())

            if self.policy_action_params:
                policy_output = self.policy.get_actions(o, ag, self.g, **self.policy_action_params)
            else:
                policy_output = self.policy.get_actions(o, ag, self.g)

            if isinstance(policy_output, np.ndarray):
                u = policy_output  # get the actions from the policy output since actions should be the first element
            else:
                u = policy_output[0]
                q = policy_output[1]
                other_histories.append(policy_output[1:])
            try:
                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
            except:
                self.logger.warn('Action "u" is not a Numpy array.')

            # visualize q value
            if self.graph:
                reset = t == 0
                # able to plot graph when compute_Q is True
                if len(policy_output) == 2:
                    self.first_env.env.add_graph_values('q-val', q ,t, reset=reset)

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
        if return_states:
            for i in range(self.rollout_batch_size):
                mj_states[i].append(self.envs[i].env.sim.get_state())

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

        if return_states:
            ret = convert_episode_to_batch_major(episode), mj_states
        else:
            ret = convert_episode_to_batch_major(episode)
        return ret


    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()

        avg_epoch_losses = []
        last_stored_idxs = []
        self.episodes_per_epoch = n_episodes

        for episode in tqdm(range(n_episodes), file=sys.__stdout__):
            # logger.info("Performing ")
            ro_start = time.time()

            episode = self.generate_rollouts()
            self.policy.store_episode(episode)

            trajectories, mj_states = super(RolloutWorker, self).generate_rollouts(return_states=True)
            stored_idxs = self.policy.store_trajectory(trajectories, mj_states=mj_states)
            last_stored_idxs = stored_idxs
            # Remove old rollouts from already replayed episodes.
            for i in stored_idxs:
                if i in self.replayed_trajectories:
                    self.replayed_trajectories.remove(i)


            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_train_batches):
                self.policy.train()

                losses = self.policy.train_model()
                if not isinstance(losses, tuple):
                    losses = [losses]
                if self.loss_histories == []:
                    self.loss_histories = [deque(maxlen=2) for _ in losses]
                if avg_epoch_losses == []:
                    avg_epoch_losses = [0 for _ in losses]
                for idx, loss in enumerate(losses):
                    avg_epoch_losses[idx] += loss

            self.policy.update_target_net()
            dur_train += time.time() - train_start

        for idx,loss in enumerate(avg_epoch_losses):
            avg_loss = (loss / n_episodes) / n_train_batches
            self.loss_histories[idx].append(avg_loss)

        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        #  TODO: Plot loss #

        return updated_policy, time_durations

    def current_mean_Q(self):
        return np.mean(self.custom_histories[0])

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics."""
        logs = []

        logs += [('success_rate', np.mean(self.success_history))]

        if self.custom_histories:
            logs += [('mean_Q', np.mean(self.custom_histories[0]))]

        logs += [('episode', self.n_episodes)]

        for i,l in enumerate(self.loss_histories):
            loss_key = 'loss-{}'.format(i)
            loss_key_map = ['observation loss', 'loss prediction loss']
            if i < len(loss_key_map):
                loss_key = loss_key_map[i]
            if len(l) > 0:
                logs += [(loss_key, l[-1])]


        return logger(logs, prefix)

