import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

import numpy as np
from baselines import logger
from baselines.template.policy import Policy

from collections import OrderedDict
# from baselines.model_based.replay_buffer import ReplayBuffer
from baselines.model_based.model_replay_buffer import ModelReplayBuffer
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class MBPolicy(Policy):
    @store_args
    def __init__(self, input_dims, model_buffer_size, model_network_class, scope, T, rollout_batch_size, **kwargs):

        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
        self.scope = scope
        self.create_model = import_function(self.model_network_class)
        #
        # # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            # self.buffer_ph_tf = [
            #     tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
        #     self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
        #
        #     self._create_network(reuse=False)
        #
        # # Configure the replay buffer.
        # input_shapes = dims_to_shapes(self.input_dims)
        # mr_buffer_shapes = {'o': (self.T + 1, *input_shapes['o']),
        #                     'u': (self.T, *input_shapes['u'])}
        #
        # self.model_replay_buffer = ModelReplayBuffer(mr_buffer_shapes, model_buffer_size)
        # pass
        print("done init MBPolicy")


    def _create_network(self, reuse=False):
        logger.info("Creating a model-based agent with action space %d x %s..." % (self.dimu, self.max_u))
        # pass
        #
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        #
        # # running averages
        # with tf.variable_scope('o_stats') as vs:
        #     if reuse:
        #         vs.reuse_variables()
        #     self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        # with tf.variable_scope('g_stats') as vs:
        #     if reuse:
        #         vs.reuse_variables()
        #     self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        #
        # # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        #
        # # networks
        with tf.variable_scope('model') as ms:
            self.model = self.create_model(batch_tf)
        # with tf.variable_scope('main') as vs:
        #     if reuse:
        #         vs.reuse_variables()
        #     self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
        #     vs.reuse_variables()
        # with tf.variable_scope('target') as vs:
        #     if reuse:
        #         vs.reuse_variables()
        #     target_batch_tf = batch_tf.copy()
        #     target_batch_tf['o'] = batch_tf['o_2']
        #     target_batch_tf['g'] = batch_tf['g_2']
        #     self.target = self.create_actor_critic(
        #         target_batch_tf, net_type='target', **self.__dict__)
        #     vs.reuse_variables()
        # assert len(self._vars("main")) == len(self._vars("target"))
        #
        # # loss functions
        obs_loss = self.model.output - batch_tf['o']
        # target_Q_pi_tf = self.target.Q_pi_tf
        # clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        # target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        # self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        # self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        # self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        # Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        # pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        # assert len(self._vars('main/Q')) == len(Q_grads_tf)
        # assert len(self._vars('main/pi')) == len(pi_grads_tf)
        # self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        # self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        # self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        # self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        #
        # # optimizers
        # self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        # self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)
        #
        # # polyak averaging
        # self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        # self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        # self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        # self.init_target_net_op = list(
        #     map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        # self.update_target_net_op = list(
        #     map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))
        #
        # # initialize all variables
        # tf.variables_initializer(self._global_vars('')).run()
        # self._sync_optimizers()
        # self._init_target_net()

    def get_actions(self, o, ag, g, policy_action_params=None):
        # This is important for the rollout (Achieved through policy). DUMMY RETURN ZEROS
        EMPTY = 0
        u = np.random.randn(o.size // self.dimo, self.dimu)
        return u, EMPTY

    def store_episode(self, episode_batch, update_stats=True):
        print("Storing episode")
        pass

    def get_current_buffer_size(self):
        print("Getting current buffer size...")
        pass

    def sample_batch(self):
        print("Sampling batch")
        pass

    def stage_batch(self, batch=None):
        print("Staging batch")
        pass

    def train(self, stage=True):
        print("Training")
        pass

    def clear_buffer(self):
        print("Clearing buffer")
        pass

    def logs(self, prefix=''):
        logs = []
        logs += [('stats/some_stat_value', 0)]
        # logs = []
        # logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        # logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        # logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        # logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

