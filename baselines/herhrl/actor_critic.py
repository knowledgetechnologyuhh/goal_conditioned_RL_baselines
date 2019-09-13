import tensorflow as tf
from baselines.util import store_args, nn
import math
import uuid

@tf.RegisterGradient("HeavisideGrad")
def _heaviside_grad(unused_op: tf.Operation, grad: tf.Tensor):
    return tf.maximum(0.0, 1.0 - tf.abs(unused_op.inputs[0])) * grad


def heaviside(x: tf.Tensor, g: tf.Graph = tf.get_default_graph()):
    custom_grads = {
        "Identity": "HeavisideGrad"
    }
    with g.gradient_override_map(custom_grads):
        i = tf.identity(x, name="identity_" + str(uuid.uuid1()))
        ge = tf.greater_equal(x, 0, name="ge_" + str(uuid.uuid1()))
        # tf.stop_gradient is needed to exclude tf.to_float from derivative
        step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
        return step_func

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
                input_pi, [self.hidden] * (self.layers+1) + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * (self.layers+1) + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * (self.layers+1) + [1], reuse=True)

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

class ActorCriticVanillaAttnEnforceW:
    steepness = 1
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
            self.attn = tf.nn.sigmoid(nn(input_og, [self.hidden] * 2 + [input_og.shape[1]], name='attn') * self.steepness)
            force_W_steepness = 3
            self.shared_preproc_err = ((4 * force_W_steepness * (self.attn ** 2)) - (4 * force_W_steepness * self.attn)) ** 2
            had_prod = self.attn * input_og
            # Now map input to a smaller space
            reduced_attn_input = had_prod
            # reduced_attn_input = nn(had_prod, [int(input_og.shape[1]//3)], name='compress_in')
            reduced_attn = self.attn
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

class ActorCriticProbSamplingAttn:
    steepness = 1
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
        # self.rnd_in_tf = inputs_tf['rnd']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_og = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('shared_preproc') as scope:
            self.prob_in = tf.nn.sigmoid(nn(input_og, [64] * 2 + [input_og.shape[1]], name='attn'))
            self.rnd = tf.random_uniform(shape=[kwargs['batch_size'], int(input_og.shape[1])])

            self.attn = tf.sigmoid((self.prob_in - self.rnd) * self.steepness)

            had_prod = self.attn * input_og
            # Now map input to a smaller space
            reduced_attn_input = had_prod
            # reduced_attn_input = nn(had_prod, [int(input_og.shape[1]//3)], name='compress_in')
            reduced_attn = self.attn
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

class ActorCriticProbSamplingAttnSteep100(ActorCriticProbSamplingAttn):
    steepness = 100
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        ActorCriticProbSamplingAttn.__init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs)

class ActorCriticProbSamplingAttnSteep6(ActorCriticProbSamplingAttn):
    steepness = 6
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        ActorCriticProbSamplingAttn.__init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs)

class ActorCriticProbSamplingAttnHeaviside:
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
            self.prob_in = tf.nn.sigmoid(nn(input_og, [64] * 2 + [input_og.shape[1]], name='attn'))
            self.rnd = tf.random_uniform(minval=0.01, maxval=0.99, shape=[kwargs['batch_size'], int(input_og.shape[1])])
            self.attn = heaviside(self.prob_in - self.rnd)

            had_prod = self.attn * input_og
            # Now map input to a smaller space
            reduced_attn_input = had_prod
            reduced_attn = self.attn
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

