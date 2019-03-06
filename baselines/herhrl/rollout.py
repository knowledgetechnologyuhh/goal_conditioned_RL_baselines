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
from tqdm import tqdm


class HierarchicalRollout(Rollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, rollout_batch_size=1,
                 exploit=False, render=False, **kwargs):

        self.is_leaf = len(policy) == 1
        this_policy = policy[0]
        dims = this_policy.input_dims
        self.T = this_policy.T
        history_len = None
        if self.is_leaf is False:
            child_policies = policy[1:]
            self.child_rollout = RolloutWorker(make_env, child_policies, dims, logger,
                                                h_level=self.h_level+1,
                                                rollout_batch_size=rollout_batch_size,
                                                render=render, **kwargs)

        # Envs are generated only at the lowest hierarchy level. Otherwise just refer to the child envs.
        if self.is_leaf is False:
            make_env = self.make_env_from_child
        self.tmp_env_ctr = 0
        Rollout.__init__(self, make_env, this_policy, dims, logger, self.T,
                         rollout_batch_size=rollout_batch_size,
                         history_len=history_len, render=render, **kwargs)

        self.env_name = self.envs[0].env.spec._env_name
        self.n_objects = self.envs[0].env.n_objects
        self.gripper_has_target = self.envs[0].env.gripper_goal != 'gripper_none'
        self.tower_height = self.envs[0].env.goal_tower_height
        self.rep_correct_history = deque(maxlen=history_len)

    def make_env_from_child(self):
        env = self.child_rollout.envs[self.tmp_env_ctr]
        self.tmp_env_ctr += 1
        return env

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

        """ ==================== Step 1: initialize s0, s1, goal =================================
        """
        if self.h_level == 0:
            self.reset_all_rollouts()

        # Setting subgoal and goal to environment, for visualization purpose
        for i, env in enumerate(self.envs):
            if self.is_leaf:
                env.env.goal = self.g[i]
            if self.h_level == 0:
                env.env.final_goal = self.g[i]
            env.env.goal_hierarchy[self.h_level] = self.g[i]


        if return_states:
            mj_states = [[] for _ in range(self.rollout_batch_size)]

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # hold custom histories throughout the iterations
        other_histories = []

        # generate episodes
        obs, achieved_goals, acts, goals, successes, penalties = [], [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key
                       in self.info_keys]

        for t_parent in tqdm(range(self.T)):
            if return_states:
                for i in range(self.rollout_batch_size):
                    mj_states[i].append(self.envs[i].env.sim.get_state())

            ''' =========================== Step 2: Sampling action a1 <-- policy pi1(s1, goal) ========================
            - if not testing: add_noise to get_action if random_sample()>0.2 otherwise get_random_action
            - if testing: get_action
            '''
            if self.policy_action_params:
                policy_output = self.policy.get_actions(o, ag, self.g, **self.policy_action_params)
            else:
                policy_output = self.policy.get_actions(o, ag, self.g)

            if isinstance(policy_output, np.ndarray):
                u = policy_output
            else:
                u = policy_output[0]
                other_histories.append(policy_output[1:])
            try:
                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)
            except:
                self.logger.warn('Action "u" is not a Numpy array.')

            # Now rescaling and offsetting u
            u = self.scale_and_offset(u)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            penalty = np.zeros(self.rollout_batch_size)
            """ ============================== Step 3: Setting subgoal g0 = subg1 <-- action a1 ========================
            """
            if self.is_leaf is False:
                self.subg = u
                self.subg = self.g.copy()  # For testing use final goal only and set n_subgoals to 1.
                self.child_rollout.g = self.subg.copy()
                self.child_rollout.generate_rollouts_update(n_episodes=1, n_train_batches=0)
            for i in range(self.rollout_batch_size):
                if self.is_leaf:
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                else:
                    curr_o_new = self.envs[i].env._get_obs()
                    # TODO: Fix penalty computation and realize penalties during training
                    penalty[i] = False
                o_new[i] = curr_o_new['observation']
                ag_new[i] = curr_o_new['achieved_goal']
                success[i] = self.envs[i].env._is_success(ag_new[i], self.g[i])
                if self.render and self.is_leaf:
                    self.envs[i].render()
                # for idx, key in enumerate(self.info_keys):
                #     info_values[idx][t_parent, i] = info[key]


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

        success_rate = np.mean(successes)
        self.success_history.append(success_rate)

        # history --> mean_Q
        if other_histories:
            for history_index in range(len(other_histories[0])):
                self.custom_histories.append(deque(maxlen=self.history_len))
                self.custom_histories[history_index].append([x[history_index] for x in other_histories])
        self.n_episodes += self.rollout_batch_size

        if return_states:
            ret = convert_episode_to_batch_major(episode), mj_states
        else:
            ret = convert_episode_to_batch_major(episode)

        """ =============== Step 8: Add penalized transition if subgoal_success doesn't have any positive value=========
        """
        return ret
    def scale_and_offset(self, u):
        if self.is_leaf:
            return u
        for i in range(self.rollout_batch_size):
            n_objects = self.envs[i].env.n_objects
            obj_height = self.envs[i].env.obj_height
            offset = np.array(list(self.envs[i].env.initial_gripper_xpos) * (n_objects + 1))
            for j, off in enumerate(offset):
                if j == 2:
                    offset[j] += self.envs[i].env.random_gripper_goal_pos_offset[2]
                elif (j+1) % 3 == 0:
                    offset[j] += obj_height * n_objects / 2
            scale_xy = self.envs[i].env.target_range
            scale_z = obj_height * n_objects / 2
            scale = np.array([scale_xy, scale_xy, scale_z] * (n_objects + 1))
            u[i] *= scale
            u[i] += offset
        return u


class RolloutWorker(HierarchicalRollout):

    @store_args
    def __init__(self, make_env, policy, dims, logger, T=None, rollout_batch_size=1,
                 exploit=False, h_level=0, render=False, **kwargs):
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
        self.h_level = h_level
        HierarchicalRollout.__init__(self, make_env, policy, dims, logger, rollout_batch_size=rollout_batch_size, render=render, **kwargs)
        self.rep_loss_history = []

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def train_policy(self, n_train_batches):
        for _ in range(n_train_batches):
            self.policy.train()  # train actor-critic
        self.policy.update_target_net()
        if self.is_leaf is False:
            self.child_rollout.train_policy(n_train_batches)

    def generate_rollouts_update(self, n_episodes, n_train_batches):
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        rep_ce_loss = 0
        self.success_history = deque(maxlen=n_episodes)
        for cyc in range(n_episodes):
            ro_start = time.time()
            episode = self.generate_rollouts()
            self.policy.store_episode(episode)


            dur_ro += time.time() - ro_start
            train_start = time.time()
            # Train only if this is the parent rollout worker
            if self.h_level == 0:
                self.train_policy(n_train_batches)
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        updated_policy = self.policy
        time_durations = (dur_total, dur_ro, dur_train)
        if n_episodes > 0 and n_train_batches > 0:
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


        logs = logger(logs, prefix+"_l_{}".format(self.h_level))

        if self.is_leaf is False:
            child_logs = self.child_rollout.logs(prefix=prefix)
            logs += child_logs

        return logs

