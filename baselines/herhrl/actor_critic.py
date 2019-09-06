import tensorflow as tf
from baselines.util import store_args, nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc'):
            self.preproc_in = input_pi
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)


class ActorCriticSharedPreproc:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc'):
            # self.preproc_in = input_pi
            self.preproc_in = nn(input_pi, [self.hidden])
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                self.preproc_in, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

class ActorCriticVanillaAttn:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_og = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc') as scope:
            attn = tf.nn.sigmoid(nn(input_og, [64] * 2 + [input_og.shape[1]], name='attn'))
            had_prod = attn * input_og
            # Now map input to a smaller space
            reduced_attn_input = had_prod
            # reduced_attn_input = nn(had_prod, [int(input_og.shape[1]//3)], name='compress_in')
            reduced_attn = attn
            # reduced_attn = nn(attn, [int(input_og.shape[1]//3)], name='compress_attn')
            self.preproc_in = tf.concat(axis=1, values=[reduced_attn_input, reduced_attn])

        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                self.preproc_in, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

class ActorCriticVanillaAttnReduced:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_og = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc') as scope:
            attn = tf.nn.sigmoid(nn(input_og, [64] * 2 + [input_og.shape[1]], name='attn'))
            had_prod = attn * input_og
            # Now map input to a smaller space
            # reduced_attn_input = had_prod
            reduced_attn_input = nn(had_prod, [int(input_og.shape[1]//3)], name='compress_in')
            # reduced_attn = attn
            reduced_attn = nn(attn, [int(input_og.shape[1]//3)], name='compress_attn')
            self.preproc_in = tf.concat(axis=1, values=[reduced_attn_input, reduced_attn])

        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                self.preproc_in, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)


class ActorCriticProbSampling:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_og = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc'):
            # rnd = tf.random_uniform(shape=[kwargs['batch_size'], int(input_og.shape[1])])
            # prob_in = tf.nn.sigmoid(nn(input_og, [64] * 2 + [input_og.shape[1]], name='attn'))
            # attn = tf.dtypes.cast(prob_in >= rnd, input_og.dtype)
            # had_prod = attn * input_og
            # # # Now map input to a smaller space
            # reduced_attn_input = had_prod
            # # # reduced_attn_input = nn(had_prod, [int(input_og.shape[1] // 3)], name='compress_in')
            # reduced_attn = attn
            # # # reduced_attn = nn(attn, [int(input_og.shape[1] // 3)], name='compress_attn')
            # self.preproc_in = tf.concat(axis=1, values=[reduced_attn_input, reduced_attn])
            self.preproc_in = input_og

        with tf.variable_scope('pi'):
            # rnd = tf.random_uniform(shape=[None, int(self.preproc_in.shape[1])])
            # batch_size = tf.placeholder(tf.int32, shape=[])  # `batch_size` is a scalar (0-D tensor).
            # rnd = tf.random_uniform([batch_size, int(self.preproc_in.shape[1])])
            # rnd = tf.random_uniform(shape=self.preproc_in.shape)
            prob_in = tf.nn.sigmoid(nn(self.preproc_in, [64] * 2 + [self.preproc_in.shape[1]], name='attn'))
            # attn = tf.dtypes.cast(prob_in >= rnd, self.preproc_in.dtype)
            # attn = prob_in * rnd
            attn = prob_in
            had_prod = attn * self.preproc_in
            reduced_attn_input = had_prod
            reduced_attn = attn
            self.pi_in = tf.concat(axis=1, values=[reduced_attn_input, reduced_attn])
            self.pi_tf = self.max_u * tf.tanh(nn(
                self.pi_in, [self.hidden] * self.layers + [self.dimu]))
        # with tf.variable_scope('pi'):
        #     self.pi_tf = self.max_u * tf.tanh(nn(
        #         self.preproc_in, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)


class ActorCriticProbSamplingReduced:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_og = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc') as scope:
            rnd = tf.random_uniform(shape=input_og.shape)
            prob_in = tf.nn.sigmoid(nn(input_og, [64] * 2 + [input_og.shape[1]], name='attn'))
            attn = tf.where(prob_in >= rnd)
            had_prod = attn * input_og
            # Now map input to a smaller space
            # reduced_attn_input = had_prod
            reduced_attn_input = nn(had_prod, [int(input_og.shape[1] // 3)], name='compress_in')
            # reduced_attn = attn
            reduced_attn = nn(attn, [int(input_og.shape[1] // 3)], name='compress_attn')
            self.preproc_in = tf.concat(axis=1, values=[reduced_attn_input, reduced_attn])

        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                self.preproc_in, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[self.preproc_in, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)