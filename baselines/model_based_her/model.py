from collections import OrderedDict

import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.util import (import_function, store_args, flatten_grads)
from baselines.common.mpi_adam import MpiAdam
from baselines.model_based_her.model_replay_buffer import ModelReplayBuffer
from baselines.template.policy import Policy


class Model(Policy):
    @store_args
    def __init__(self,input_dims, T, rollout_batch_size, max_u, scope,  **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
        self.create_predictor_model = import_function(self.model_network_class)
        model_shapes = OrderedDict()
        time_dim = T
        self.scope = scope
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

        #  TODO: Pass from click args #
        buff_sampling = 'random'
        memval_method = 'uniform'
        self.model_replay_buffer = ModelReplayBuffer(self.model_shapes, self.model_buffer_size, buff_sampling, memval_method)

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def store_trajectory(self, episode, update_stats=True, mj_states=None):
        trajectories = []
        for e_idx in range(len(episode['o'])):
            traj = {}
            traj['o'] = episode['o'][e_idx][:-1]
            traj['u'] = episode['u'][e_idx]
            traj['o2'] = episode['o'][e_idx][1:]
            trajectories.append(traj)

        new_idxs = self.model_replay_buffer.store_episode(trajectories, mj_states)
        self.update_replay_buffer_losses(new_idxs)
        return new_idxs

    def sample_trajectory_batch(self, batch_size=None, idxs=None):
        if idxs is not None:
            batch_dict, idxs = self.model_replay_buffer.get_rollouts_by_idx(idxs)
        else:
            if batch_size is None:
                batch_size = self.model_train_batch_size
            batch_dict, idxs = self.model_replay_buffer.sample(batch_size)
        batch = [batch_dict[key] for key in self.model_shapes.keys()]
        return batch, idxs

    def _sync_optimizers(self):
        self.obs_pred_adam.sync()
        self.loss_pred_adam.sync()

    def _create_predictor_network(self, sess):
        self.sess = sess
        logger.info("Creating a model-based agent with action space %d x %s..." % (self.dimu, self.max_u))

        # mini-batch sampling.
        model_batch_tf = OrderedDict([(key, self.model_buffer_ph_tf[i])
                                      for i, key in enumerate(self.model_shapes.keys())])
        # networks
        with tf.variable_scope('model') as ms:
            self.prediction_model = self.create_predictor_model(model_batch_tf, **self.__dict__)
            ms.reuse_variables()

        model_grads = tf.gradients(self.prediction_model.obs_loss_tf, self._vars('model/ModelRNN'))
        self.model_grads_tf = flatten_grads(grads=model_grads, var_list=self._vars('model/ModelRNN'))

        # optimizers
        self.obs_pred_adam = MpiAdam(self._vars('model/ModelRNN'), scale_grad_by_procs=False)
        self.loss_pred_adam = MpiAdam(self._vars('model/LossPredNN'), scale_grad_by_procs=False)

        obs_model_grads = tf.gradients(self.prediction_model.obs_loss_tf, self._vars('model/ModelRNN'))
        self.obs_model_grads_tf = flatten_grads(grads=obs_model_grads, var_list=self._vars('model/ModelRNN'))

        loss_model_grads = tf.gradients(self.prediction_model.loss_loss_tf, self._vars('model/LossPredNN'))
        self.loss_model_grads_tf = flatten_grads(grads=loss_model_grads, var_list=self._vars('model/LossPredNN'))

    def update_replay_buffer_losses(self, buffer_idxs):
        batch_size_diff = self.model_train_batch_size - len(buffer_idxs)
        if batch_size_diff < 0:
            print("ERROR!!! cannot predict more episodes than model_train_buffer_size")
            assert False
        padded_buffer_idxs = buffer_idxs + [0] * batch_size_diff
        batch, idxs = self.sample_trajectory_batch(idxs=padded_buffer_idxs)
        obs_loss_per_step, loss_loss_per_step, loss_pred_per_step, obs_model_grads, loss_model_grads = self.get_model_grads(batch)
        self.model_replay_buffer.update_with_loss(buffer_idxs, obs_loss_per_step[:len(buffer_idxs)], loss_pred_per_step[:len(buffer_idxs)])



    def forward_step(self, u, o, s):

        bs = self.model_train_batch_size
        o_pad = bs - len(o)
        u_pad = bs - len(u)
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


    def get_actions_max_surprise(self, o, pred_lookahead=15):
        u_s = []

        for rollout_idx in range(self.rollout_batch_size):
            max_l_branch = np.finfo(np.float16).min
            state = None
            obs = np.array([np.array([o[rollout_idx]]) for _ in range(self.model_train_batch_size)])
            current_seq = []
            next_u = np.random.rand(self.model_train_batch_size, 1, self.dimu) * 2 - 1

            for step in range(pred_lookahead):
                u = np.random.rand(self.model_train_batch_size, 1, self.dimu) * 2 - 1
                current_seq.append(u)
                obs, loss, state = self.forward_step(u, obs, state)
                # loss as indicator for novelty of this trajectory
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

    def _update_model(self, model_grads, loss_grads):
        self.obs_pred_adam.update(model_grads, self.model_lr)
        self.loss_pred_adam.update(loss_grads, self.model_lr)

    def get_model_grads(self, batch):
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

    def train_model(self, stage=False):
        batch, buffer_idxs = self.sample_trajectory_batch()
        obs_loss_per_step, loss_loss_per_step, loss_pred_per_step, obs_model_grads, loss_model_grads = self.get_model_grads(batch)
        mean_obs_loss = np.mean(obs_loss_per_step)
        mean_loss_loss = np.mean(loss_loss_per_step)
        self.model_replay_buffer.update_with_loss(buffer_idxs, obs_loss_per_step, loss_pred_per_step)
        self._update_model(obs_model_grads, loss_model_grads)
        return mean_obs_loss, mean_loss_loss

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res


    def __getstate__(self):
        """Our model can be loaded from pkl, but after unpickling you cannot continue training. """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'model_replay_buffer', 'sess', '_stats',
                             'prediction_model', 'lock', 'stage_shapes', 'model_shapes', 'create_predictor_model']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['model_buffer_size'] = self.model_buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'model_replay_buffer' not in x.name])
        return state




