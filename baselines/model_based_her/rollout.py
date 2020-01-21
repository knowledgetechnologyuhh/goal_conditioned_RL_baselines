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

        self.replayed_episodes = []
        self.loss_histories = []

        self.pred_err = None
        self.pred_accumulated_err = None
        self.pred_err_std = None
        self.pred_steps = None
        self.mj_pred_err = None
        self.mj_pred_accumulated_err = None
        self.mj_pred_err_std = None
        self.mj_pred_steps = None
        self.loss_pred_steps = None
        self.buff_obs_variance = None
        self.surprise_fig = None
        self.top_exp_replay_values = deque(maxlen=10)
        self.episodes_per_epoch = None
        self.visualize_replay = False
        self.record_replay = True
        self.do_plot = False
        self.test_mujoco_err = True
        self.acc_err_steps = 5 # Number of steps to accumulate the prediction error


    def generate_rollouts(self, return_states=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
           policy acting on it accordingly. """

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

            # use method of parent class
            episode = super(RolloutWorker, self).generate_rollouts()
            self.policy.store_episode(episode)

            # use custom generate_rollouts method
            rollouts, mj_states = self.generate_rollouts(return_states=True)
            stored_idxs = self.policy.model.store_trajectory(rollouts, mj_states=mj_states)
            last_stored_idxs = stored_idxs

            # Remove old rollouts from already replayed episodes.
            for i in stored_idxs:
                if i in self.replayed_episodes:
                    self.replayed_episodes.remove(i)

            dur_ro += time.time() - ro_start
            dur_train += self.train_models(n_episodes, n_train_batches, avg_epoch_losses)

        self.buff_obs_variance = self.policy.model.model_replay_buffer.get_variance(column='o')
        print('VAR', self.buff_obs_variance)
        self.test_prediction_error(last_stored_idxs)

        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        #  TODO: Plot loss #

        return updated_policy, time_durations

    def train_models(self, n_episodes, n_train_batches, avg_epoch_losses):
        """train policy and forward model"""
        train_start = time.time()

        for _ in range(n_train_batches):
            self.policy.train()
            losses = self.policy.model.train_model()

            if not isinstance(losses, tuple):
                losses = [losses]
            if self.loss_histories == []:
                self.loss_histories = [deque(maxlen=2) for _ in losses]
            if avg_epoch_losses == []:
                avg_epoch_losses = [0 for _ in losses]
            for idx, loss in enumerate(losses):
                avg_epoch_losses[idx] += loss

        dur_train = time.time() - train_start
        self.policy.update_target_net()

        for idx,loss in enumerate(avg_epoch_losses):
            avg_loss = (loss / n_episodes) / n_train_batches
            self.loss_histories[idx].append(avg_loss)

        return dur_train

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

        if self.buff_obs_variance is not None:
            logs += [('buffer observation variance', self.buff_obs_variance)]



        return logger(logs, prefix)

    def test_prediction_error(self, buffer_idxs):

        this_pred_err_hist = []
        this_accumulated_pred_err = []
        this_pred_std_hist = []
        this_mj_pred_err_hist = []
        this_mj_accumulated_pred_err = []
        this_mj_pred_std_hist = []

        loss_pred_threshold_perc = 20 # Loss prediction threshold in %
        batch, idxs = self.policy.model.sample_trajectory_batch(idxs=buffer_idxs)
        s = None
        o_s_batch, o2_s_batch, u_s_batch, loss_s_batch, loss_pred_s_batch = batch[0], batch[1], batch[2], batch[3], batch[4]

        # Measure mean prediction error
        for step in range(self.T):
            o_s_step, o2_s_step, u_s_step = \
                o_s_batch[:, step:step + 1, :], o2_s_batch[:, step:step + 1, :], \
                u_s_batch[:, step:step + 1, :]
            o2_pred, l_s, s = self.policy.model.forward_step(u_s_step, o_s_step, s)
            err = np.abs(o2_s_step - o2_pred)
            ep_err_mean_step = np.mean(err,axis=2)
            ep_err_mean = np.mean(ep_err_mean_step)
            ep_err_std = np.std(err)
            this_pred_err_hist.append(ep_err_mean)
            this_pred_std_hist.append(ep_err_std)

        # Test mujoco prediction error as baseline
        if self.test_mujoco_err:
            for step in range(self.T):
                o_s_step, o2_s_step, u_s_step = \
                    o_s_batch[:, step, :], o2_s_batch[:, step, :], \
                    u_s_batch[:, step, :]
                assert len(self.envs) == len(buffer_idxs) # Number of parallel rollout environment instances must be equal to to batch size.
                mj_states = [self.policy.model.model_replay_buffer.mj_states[buff_idx][step] for buff_idx in buffer_idxs]
                [self.envs[i].env.sim.set_state(mj_state) for i, mj_state in enumerate(mj_states)]
                [self.envs[i].env.sim.forward() for i, mj_state in enumerate(mj_states)]
                o_pred = np.array([self.envs[i].env._get_obs()['observation'] for i in range(len(buffer_idxs))])

                # TODO: For debugging... tmp_err should be 0 or very close to 0, as we are just reloading the mujoco state that corresponds to the observation.
                # TODO: However, tt is larger than expected. Specifically, it is larger that the accumulated mujoco error computed below.
                # TODO: This should not be the case.
                tmp_err = np.abs(o_s_step - o_pred)
                tmp_err_mean = np.mean(tmp_err)

                [self.envs[i].env.step(u) for i, u in enumerate(u_s_step)]
                o2_pred = np.array([self.envs[i].env._get_obs()['observation'] for i in range(len(buffer_idxs))])
                err = np.abs(o2_s_step - o2_pred)
                ep_err_mean = np.mean(err)
                ep_err_std = np.std(err)
                this_mj_pred_err_hist.append(ep_err_mean)
                this_mj_pred_std_hist.append(ep_err_std)


        # Measure the number of steps that can be forward propagated until the observation error gets larger than the goal achievement threshold.
        initial_o_s = o_s_batch[:,0:1,:]
        s = None
        this_pred_steps = [self.T for _ in initial_o_s]
        this_mj_pred_steps = [self.T for _ in initial_o_s]
        this_loss_pred_steps = [self.T for _ in initial_o_s]
        o_s_step = initial_o_s
        for step in range(self.T):
            o2_s_step, u_s_step = o2_s_batch[:, step:step + 1, :], u_s_batch[:, step:step + 1, :]
            o_s_step, l_s, s = self.policy.model.forward_step(u_s_step, o_s_step, s)
            flattened_o_s_step = o_s_step[:, 0,:]
            flattened_o2_s_step = o2_s_step[:, 0, :]
            fwd_goal_g = self.policy.model.env._obs2goal(flattened_o_s_step)
            orig_g = self.policy.model.env._obs2goal(flattened_o2_s_step)
            pred_successes = self.policy.model.env._is_success(fwd_goal_g, orig_g)
            for batch_idx, pred_success in enumerate(pred_successes):
                if not pred_success and this_pred_steps[batch_idx] == self.T:
                    this_pred_steps[batch_idx] = step
                loss_pred_ok = loss_s_batch[batch_idx][step] + (loss_s_batch[batch_idx][step] * loss_pred_threshold_perc / 100) < l_s[batch_idx][0]
                if not loss_pred_ok and this_loss_pred_steps[batch_idx] == self.T:
                    this_loss_pred_steps[batch_idx] = step
            if step == self.acc_err_steps:
                this_accumulated_pred_err = np.abs(flattened_o_s_step - flattened_o2_s_step)

        # TODO: The accumulated mujoco error is smaller than the per-step mujoco error computed above. Also,
        # TODO: Mujoco seems to always be able to reproduce all steps. This should also not be the case.
        if self.test_mujoco_err:
            assert len(self.envs) == len(buffer_idxs)  # Number of parallel rollout environment instances must be equal to to batch size.
            init_mj_states = [self.policy.model.model_replay_buffer.mj_states[buff_idx][0] for buff_idx in buffer_idxs]
            [self.envs[i].env.sim.set_state(mj_state) for i, mj_state in enumerate(init_mj_states)]
            [self.envs[i].env.sim.forward() for i, mj_state in enumerate(mj_states)]
            for step in range(self.T):
                o2_s_step, u_s_step = o2_s_batch[:, step, :], u_s_batch[:, step, :]
                [self.envs[i].env.step(u) for i, u in enumerate(u_s_step)]
                o_s_step = np.array([self.envs[i].env._get_obs()['observation'] for i in range(len(buffer_idxs))])
                fwd_goal_g = self.policy.model.env._obs2goal(o_s_step)
                orig_g = self.policy.model.env._obs2goal(o2_s_step)
                pred_successes = self.policy.model.env._is_success(fwd_goal_g, orig_g)
                err = np.abs(o_s_step - o2_s_step)
                tmp_mean_err = np.mean(err)
                # print(tmp_mean_err)
                for batch_idx, pred_success in enumerate(pred_successes):
                    if not pred_success and this_pred_steps[batch_idx] == self.T:
                        this_mj_pred_steps[batch_idx] = step
                if step == self.acc_err_steps:
                    this_mj_accumulated_pred_err = err

        pred_err_mean = np.mean(this_pred_err_hist)
        pred_err_std_mean = np.mean(this_pred_std_hist)
        pred_steps_mean = np.mean(this_pred_steps)
        this_accumulated_pred_err_mean = np.mean(this_accumulated_pred_err)
        loss_pred_steps_mean = np.mean(this_loss_pred_steps)
        self.pred_err = pred_err_mean
        self.pred_err_std = pred_err_std_mean
        self.pred_steps = pred_steps_mean
        self.pred_accumulated_err = this_accumulated_pred_err_mean
        self.loss_pred_steps = loss_pred_steps_mean
        self.policy.model.loss_pred_reliable_fwd_steps = self.loss_pred_steps

        if self.test_mujoco_err:
            mj_pred_err_mean = np.mean(this_mj_pred_err_hist)
            mj_pred_err_std_mean = np.mean(this_mj_pred_std_hist)
            mj_pred_steps_mean = np.mean(this_mj_pred_steps)
            mj_accumulated_pred_err_mean = np.mean(this_mj_accumulated_pred_err)
            self.mj_pred_err = mj_pred_err_mean
            self.mj_pred_err_std = mj_pred_err_std_mean
            self.mj_pred_steps = mj_pred_steps_mean
            self.mj_pred_accumulated_err = mj_accumulated_pred_err_mean


