import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
import collections
import numpy as np
from baselines import logger
from baselines.template.policy import Policy
from baselines.common.mpi_adam import MpiAdam

from collections import OrderedDict
# from baselines.model_based.replay_buffer import ReplayBuffer
from baselines.model_based.model_replay_buffer import ModelReplayBuffer
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class MBPolicy(Policy):
    @store_args
    def __init__(self, input_dims, model_buffer_size, model_network_class, scope, T, rollout_batch_size, model_lr, model_train_batch_size, buff_sampling, memval_method, **kwargs):

        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
        self.env = kwargs['env']
        self.current_fwd_step_hist = []
        self.scope = scope
        self.create_model = import_function(model_network_class)
        #
        # Create network.
        model_shapes = OrderedDict()
        time_dim = self.T
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            if key in ['o']:
                model_shapes[key] = (None, time_dim, *self.input_shapes[key])
                model_shapes[key +"2"] = (None, time_dim, *self.input_shapes[key])
            if key in ['u']:
                model_shapes[key] = (None, time_dim, *self.input_shapes[key])

        # Add loss and loss prediction to model
        model_shapes['loss'] = (None, time_dim, 1)
        model_shapes['loss_pred'] = (None, time_dim, 1)

        self.model_shapes = model_shapes

        with tf.variable_scope(self.scope):
            self.model_buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=(None, None, shape[2])) for _, shape in self.model_shapes.items()]
            self._create_network(reuse=False)

        # Initialize the model replay buffer.
        self.model_replay_buffer = ModelReplayBuffer(self.model_shapes, model_buffer_size, buff_sampling, memval_method)
        self.loss_pred_reliable_fwd_steps = 0
        print("done init MBPolicy")


    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def get_init_zero_state(self):
        pred_net2 = self.create_model(
            {'o': tf.placeholder(tf.float32, shape=(None, None, 1)), 'o2': tf.placeholder(tf.float32, shape=(None, None, 1)), 'u': tf.placeholder(tf.float32, shape=(None, None, 1))},
            **self.__dict__)
        if 'initial_state' in pred_net2.__dict__.keys():
            return pred_net2.initial_state
        else:
            return None

    def _create_network(self, reuse=False):
        logger.info("Creating a model-based agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # # mini-batch sampling.
        model_batch_tf = OrderedDict([(key, self.model_buffer_ph_tf[i])
                                      for i, key in enumerate(self.model_shapes.keys())])
        # # networks
        with tf.variable_scope('model') as ms:
            self.prediction_model = self.create_model(model_batch_tf, **self.__dict__)
            ms.reuse_variables()

        model_grads = tf.gradients(self.prediction_model.obs_loss_tf, self._vars('model/ModelRNN'))
        self.model_grads_tf = flatten_grads(grads=model_grads, var_list=self._vars('model/ModelRNN'))

        # # optimizers
        self.obs_pred_adam = MpiAdam(self._vars('model/ModelRNN'), scale_grad_by_procs=False)
        self.loss_pred_adam = MpiAdam(self._vars('model/LossPredNN'), scale_grad_by_procs=False)

        obs_model_grads = tf.gradients(self.prediction_model.obs_loss_tf, self._vars('model/ModelRNN'))
        self.obs_model_grads_tf = flatten_grads(grads=obs_model_grads, var_list=self._vars('model/ModelRNN'))

        loss_model_grads = tf.gradients(self.prediction_model.loss_loss_tf, self._vars('model/LossPredNN'))
        self.loss_model_grads_tf = flatten_grads(grads=loss_model_grads, var_list=self._vars('model/LossPredNN'))

        # # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()

    def get_actions(self, o, ag, g, policy_action_params=None):
        if self.action_selection == 'random':
            u = self.get_actions_random()
        elif self.action_selection == 'max_pred_surprise':
            u = self.get_actions_max_surprise(o)
        return u

    def get_actions_random(self):
        u_s = []
        for ro_idx in range(self.rollout_batch_size):
            u = np.random.randn(self.dimu)
            u_s.append(u)
        u_s = np.array(u_s)
        if len(u_s) == 1:
            return u_s[0]
        else:
            return u_s

    def get_actions_max_surprise(self, o, pred_lookahead=15):
        u_s = []
        for ro_idx in range(self.rollout_batch_size):
            max_l_branch = np.finfo(np.float16).min
            state = None
            obs = np.array([np.array([o[ro_idx]]) for _ in range(self.model_train_batch_size)])
            current_seq = []
            next_u = np.random.rand(self.model_train_batch_size, 1, self.dimu) * 2 - 1
            for step in range(pred_lookahead):
                u = np.random.rand(self.model_train_batch_size, 1, self.dimu) * 2 - 1
                current_seq.append(u)
                obs, loss, state = self.forward_step(u, obs, state)
                if np.max(loss) > max_l_branch:
                    next_u = current_seq[0][np.argmax(loss)][0]
                    max_l_branch = np.max(loss)
                current_seq.append(u)

            u_s.append(next_u)
        u_s = np.array(u_s)
        if len(u_s) == 1:
            return u_s[0]
        else:
            return u_s

    def get_actions_policy(self, o, ag, g, policy_action_params=None):
        batch = self.model_replay_buffer.sample(self.model_replay_buffer.current_size)
        ep_len = batch['o2'].shape[1]
        ep_steps_remaining = ep_len - self.env.step_ctr
        u_s = []
        for ro_idx in range(self.rollout_batch_size):
            obs_goal = self.env._obs2goal(o[ro_idx])
            if self.env._is_success(g[ro_idx], obs_goal):
                u = np.zeros(self.dimu)

            elif True:
                # print("Using model and experience to find appropriate action towards achieving the goal")

                smallest_dist_to_goal = np.finfo(np.float32).max
                smallest_dist_to_goal_obs_idx = -1
                smallest_dist_to_goal_obs_seq_idx = -1
                smallest_dist_to_obs = np.finfo(np.float32).max
                smallest_dist_to_obs_obs_idx = -1
                smallest_dist_to_obs_obs_seq_idx = -1

                # Find sequence with state that is closest to goal and sequence with state that is closest to observation.
                for obs_seq_idx, obs_seq in enumerate(batch['o2']):
                    for obs_idx, obs in enumerate(obs_seq):
                        obs_goal = self.env._obs2goal(obs)
                        goal_dist = np.linalg.norm(obs_goal - g[ro_idx], axis=-1)
                        if goal_dist < smallest_dist_to_goal:
                            smallest_dist_to_goal = goal_dist
                            smallest_dist_to_goal_obs_idx = obs_idx
                            smallest_dist_to_goal_obs_seq_idx = obs_seq_idx
                        if (obs_idx + 1) < ep_len:
                            obs_dist = np.linalg.norm(obs - o[ro_idx], axis=-1)
                            if obs_dist < smallest_dist_to_obs:
                                smallest_dist_to_obs = obs_dist
                                smallest_dist_to_obs_obs_idx = obs_idx + 1
                                smallest_dist_to_obs_obs_seq_idx = obs_seq_idx

                # Now connect both sequences at minimal intersection point, such that the resulting sequence contains at most ep_steps_remaining transitions
                smallest_obs_dist = np.finfo(np.float32).max
                intersection_idxs = (smallest_dist_to_obs_obs_idx, smallest_dist_to_goal_obs_idx)
                for og_idx, obs_goal in enumerate(batch['o2'][smallest_dist_to_goal_obs_seq_idx][:smallest_dist_to_goal_obs_idx]):
                    inter_to_goal_steps = smallest_dist_to_goal_obs_idx - og_idx
                    for oo_idx, obs_obs in enumerate(batch['o'][smallest_dist_to_obs_obs_seq_idx][smallest_dist_to_obs_obs_idx:]):
                        obs_to_inter_steps = oo_idx
                        total_steps = inter_to_goal_steps + obs_to_inter_steps
                        if total_steps > ep_steps_remaining:
                            continue
                        this_dist = np.linalg.norm(obs_goal - obs_obs, axis=-1)
                        if this_dist < smallest_obs_dist:
                            intersection_idxs = (oo_idx + smallest_dist_to_obs_obs_idx, og_idx)
                            smallest_obs_dist = this_dist

                actions = np.concatenate(
                    (batch['u'][smallest_dist_to_obs_obs_seq_idx][smallest_dist_to_obs_obs_idx:intersection_idxs[0]],
                     batch['u'][smallest_dist_to_goal_obs_seq_idx][intersection_idxs[1]:smallest_dist_to_goal_obs_idx]))

                success = False
                next_o = o[ro_idx]
                next_s = None # The optional internal state in case of using RNNs.
                for u in actions:
                    next_o, next_s = self.forward_step_single(u,next_o, next_s)
                    next_o_goal = self.env._obs2goal(next_o)
                    if self.env._is_success(g[ro_idx], next_o_goal):
                        u = actions[0]
                        # print("I found a plan that should work, according to the learned forward model.")
                        success = True
                        break
                if success is False:
                    u = np.random.randn(self.dimu)
            else:
                # print("Selecting action that maximizes surprisal-based exploration")
                u = np.random.randn(self.dimu)
            u_s.append(u)
        u_s = np.array(u_s)
        if len(u_s) == 1:
            return u_s[0]
        else:
            return u_s

    def update_replay_buffer_losses(self, buffer_idxs):
        batch_size_diff = self.model_train_batch_size - len(buffer_idxs)
        if batch_size_diff < 0:
            print("ERROR!!! cannot predict more episodes than model_train_buffer_size")
            assert False
        padded_buffer_idxs = buffer_idxs + [0] * batch_size_diff
        batch, idxs = self.sample_batch(idxs=padded_buffer_idxs)
        obs_loss_per_step, loss_loss_per_step, loss_pred_per_step, obs_model_grads, loss_model_grads = self.get_grads(batch)
        self.model_replay_buffer.update_with_loss(buffer_idxs, obs_loss_per_step[:len(buffer_idxs)], loss_pred_per_step[:len(buffer_idxs)])
        pass

    def store_episode(self, episode, update_stats=True, mj_states=None):
        rollouts = []
        for e_idx in range(len(episode['o'])):
            rollout = {}
            rollout['o'] = episode['o'][e_idx][:-1]
            rollout['u'] = episode['u'][e_idx]
            rollout['o2'] = episode['o'][e_idx][1:]
            rollouts.append(rollout)

        new_idxs = self.model_replay_buffer.store_episode(rollouts, mj_states)
        self.update_replay_buffer_losses(new_idxs)
        return new_idxs
    
    def sample_batch(self, batch_size=None, idxs=None):
        if idxs is not None:
            batch_dict, idxs = self.model_replay_buffer.get_rollouts_by_idx(idxs)
        else:
            if batch_size is None:
                batch_size = self.model_train_batch_size
            batch_dict, idxs = self.model_replay_buffer.sample(batch_size)
        batch = [batch_dict[key] for key in self.model_shapes.keys()]
        return batch, idxs

    def forward_step_single(self, u, o, s):
        # TODO: Use function forward_step instead.
        next_o, pred_l, s2 = self.forward_step(np.array([np.array([u])]), np.array([np.array([o])]), s)
        next_o = next_o[0][0]
        pred_l = pred_l[0][0]
        return next_o, pred_l, s2

    def forward_step(self, u, o, s):

        bs = self.model_train_batch_size
        o_pad = bs - len(o)
        # o_pad_vals = np.zeros(o.shape[1:])
        u_pad = bs - len(u)
        # u_pad_vals = np.zeros(u.shape[1:])
        padded_o = np.pad(o, ((0, o_pad),(0,0),(0,0)), 'constant', constant_values=0)
        padded_u = np.pad(u, ((0, u_pad),(0,0),(0,0)), 'constant', constant_values=0)
        if s is not None:
            padded_s = np.pad(s, (0,bs - len(s)), 'constant', constant_values=0)
        else:
            padded_s = None

        step_batch = [padded_o, padded_u]

        fd = {self.prediction_model.o_tf: step_batch[0], self.prediction_model.u_tf: step_batch[1]}
        if s is not None and 'initial_state' in self.prediction_model.__dict__:
            fd[self.prediction_model.initial_state] = padded_s

        fetches = [self.prediction_model.output, self.prediction_model.loss_prediction_tf]
        if 'rnn_state' in self.prediction_model.__dict__:
            fetches.append(self.prediction_model.rnn_state)
            o2, l, s2 = self.sess.run(fetches, feed_dict=fd)
        else:
            o2, l = np.array(self.sess.run(fetches, feed_dict=fd)[0])
            s2 = None
        next_o = o2[:len(o)]
        pred_l = l[:len(o)]
        return next_o, pred_l, s2

    def _update(self, model_grads, loss_grads):
        self.obs_pred_adam.update(model_grads, self.model_lr)
        self.loss_pred_adam.update(loss_grads, self.model_lr)

    def get_grads(self, batch):
        fetches = [
            self.prediction_model.obs_loss_per_step_tf,
            self.prediction_model.loss_loss_per_step_tf,
            self.prediction_model.loss_prediction_tf,
            self.obs_model_grads_tf,
            self.loss_model_grads_tf
        ]
        fd = {self.prediction_model.o_tf: batch[0],
              self.prediction_model.o2_tf: batch[1],
              self.prediction_model.u_tf: batch[2],
              self.prediction_model.loss_tf: batch[3]}
        obs_loss_per_step, loss_loss_per_step, loss_pred_per_step, obs_model_grads, loss_model_grads = self.sess.run(fetches, fd)
        return obs_loss_per_step, loss_loss_per_step, loss_pred_per_step, obs_model_grads, loss_model_grads


    def train(self, stage=False):
        batch, buffer_idxs = self.sample_batch()
        obs_loss_per_step, loss_loss_per_step, loss_pred_per_step, obs_model_grads, loss_model_grads = self.get_grads(batch)
        mean_obs_loss = np.mean(obs_loss_per_step)
        mean_loss_loss = np.mean(loss_loss_per_step)
        self.model_replay_buffer.update_with_loss(buffer_idxs, obs_loss_per_step, loss_pred_per_step)
        self._update(obs_model_grads, loss_model_grads)
        return mean_obs_loss, mean_loss_loss

    def logs(self, prefix=''):
        logs = []
        logs += [('mb_policy/model_lr', self.model_lr)]
        # logs = []
        # logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        # logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        # logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        # logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _sync_optimizers(self):
        self.obs_pred_adam.sync()
        self.loss_pred_adam.sync()

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        # [print(key, ": ", item) for key, item in self.__dict__.items()]
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'model_replay_buffer', 'sess', '_stats',
                             'prediction_model', 'lock',
                             'stage_shapes', 'model_shapes', 'create_model']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['model_buffer_size'] = self.model_buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'model_replay_buffer' not in x.name])
        return state

    def __setstate__(self, state):
        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'model_replay_buffer' not in x.name]
        assert (len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

