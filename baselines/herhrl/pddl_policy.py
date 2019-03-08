from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, prob_dist2discrete)
from baselines.herhrl.normalizer import Normalizer
from baselines.herhrl.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.herhrl.hrl_policy import HRL_Policy
from baselines.herhrl.obs2preds import Obs2PredsModel, Obs2PredsBuffer
# from baselines.her_pddl.pddl.pddl_util import obs_to_preds_single



def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class PDDL_POLICY(HRL_Policy):
    @store_args
    def __init__(self, input_dims, max_u, T, rollout_batch_size, **kwargs):
        """Implementation of PDDL-Planner that generates actions and grounds them to low-level subgoals.
        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer, reward function is called here
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        HRL_Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        self.max_u = max_u

    # def _random_action(self, n):
    #     return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False, exploit=True):

        u = []
        for i in range(self.rollout_batch_size):
            # self.envs[i].env.final_goal = g[i] #TODO: Check why goal has to be set here. It should be set already by reference during the rollout.
            # last_n_hots.append(self.envs[i].env.get_preds()[1])
            plan = self.envs[i].env.get_plan()
            if len(plan[0]) > 0:
                this_u = self.envs[i].env.action2subgoal(plan[0][0])
            else:
                this_u = g[i].copy()
            u.append(this_u)
        u = np.array(u)

        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()

        return u

    def store_episode(self, episode_batch, update_stats=True):
        pass

    def train(self, stage=True):
        pass

    def update_target_net(self):
        pass

    # def __getstate__(self):
    #     excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
    #                          'main', 'target', 'lock', 'env', 'sample_transitions',
    #                          'stage_shapes', 'create_actor_critic',
    #                          'obs2preds_buffer', 'obs2preds_model']
    #
    #     state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
    #     state['buffer_size'] = self.buffer_size
    #     state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name and 'obs2preds_buffer' not in x.name])
    #     return state
    #
    # def __setstate__(self, state):
    #     if 'sample_transitions' not in state:
    #         # We don't need this for playing the policy.
    #         state['sample_transitions'] = None
    #
    #     self.__init__(**state)
    #     # set up stats (they are overwritten in __init__)
    #     for k, v in state.items():
    #         if k[-6:] == '_stats':
    #             self.__dict__[k] = v
    #     # load TF variables
    #     vars = [x for x in self._global_vars('') if 'buffer' not in x.name and 'obs2preds_buffer' not in x.name]
    #     assert(len(vars) == len(state["tf"]))
    #     node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
    #     self.sess.run(node)
