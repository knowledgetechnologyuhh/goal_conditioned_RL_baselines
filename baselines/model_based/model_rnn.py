import tensorflow as tf
from baselines.util import store_args, nn


class ModelRNN:

    @store_args
    def __init__(self, inputs_tf):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o) and the action (u)
            dimo (int): the dimension of the observations
            dimu (int): the dimension of the actions
            layer_sizes ([int, ...., int]): number of hidden units for each layer
        """
        print("Initializing model")
        self.o_tf = inputs_tf['o']
        self.u_tf = inputs_tf['u']
        #
        dimo = self.o_tf.shape[1]
        #
        hidden = 100
        layers = 3
        #
        with tf.variable_scope('ModelRNN'):
            input = tf.concat(axis=1, values=[self.o_tf, self.u_tf])
            self.output = nn(input, [hidden] * layers + [dimo])

            # # for policy training
            # input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            # self.Q_pi_tf = nn(input_Q, [hidden] * layers + [1])
            # # for critic training
            # input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            # self._input_Q = input_Q  # exposed for tests
            # self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)
