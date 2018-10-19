import tensorflow as tf
from baselines.util import store_args, nn
from tensorflow.contrib.rnn import GRUCell, LSTMCell
import numpy as np

# For resetable/resumable LSTMs look here:
# https://stackoverflow.com/questions/38441589/is-rnn-initial-state-reset-for-subsequent-mini-batches
# https://stackoverflow.com/questions/48523923/how-to-reset-the-state-of-a-gru-in-tensorflow-after-every-epoch


class Resetable_GRU:
    @store_args
    def __init__(self, inputs_tf, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
        """
        print("Initializing model")
        self.o_tf = inputs_tf['o']
        self.o2_tf = inputs_tf['o2']
        self.u_tf = inputs_tf['u']
        self.max_batch_size = kwargs['model_train_batch_size']

        #
        dimo = self.o_tf.shape[2]
        #
        size = 100

        # layers = 3
        with tf.variable_scope('ModelRNN'):
            # create a BasicRNNCell
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(size)

            # defining initial zero state
            self.initial_state = self.rnn_cell.zero_state(self.max_batch_size, dtype=tf.float32)

            input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])

            out, state = tf.nn.dynamic_rnn(
                self.rnn_cell,
                input,
                initial_state=self.initial_state,
                dtype=tf.float32)

            self.state = state
            self.output = tf.layers.dense(out, dimo)

        # self.obs_loss_tf = tf.reduce_mean(tf.square(self.output - self.o2_tf))

class GRU_Simple:
    @store_args
    def __init__(self, inputs_tf, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
        """
        print("Initializing model")
        self.o_tf = inputs_tf['o']
        self.o2_tf = inputs_tf['o2']
        self.u_tf = inputs_tf['u']
        self.max_batch_size = kwargs['model_train_batch_size']

        #
        dimo = self.o_tf.shape[2]
        #
        size = 100

        # layers = 3
        with tf.variable_scope('ModelRNN'):
            # create a BasicRNNCell
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(size)

            input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])

            out, state = tf.nn.dynamic_rnn(
                self.rnn_cell,
                input,
                dtype=tf.float32)

            self.state = state
            self.output = tf.layers.dense(out, dimo)

        self.obs_loss_tf = tf.reduce_mean(tf.square(self.output - self.o2_tf))


class LSTM_Simple:
    @store_args
    def __init__(self, inputs_tf, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
        """
        print("Initializing model")
        self.o_tf = inputs_tf['o']
        self.u_tf = inputs_tf['u']
        #
        dimo = self.o_tf.shape[2]
        #
        size = 100
        # layers = 3
        #
        with tf.variable_scope('ModelRNN'):
            input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])
            out, states = tf.nn.dynamic_rnn(
                LSTMCell(size),
                input,
                dtype=tf.float32)

            self.output = tf.layers.dense(out, dimo)


class FF_Simple:
    @store_args
    def __init__(self, inputs_tf, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
        """
        print("Initializing model")
        self.o_tf = inputs_tf['o']
        self.u_tf = inputs_tf['u']
        #
        dimo = self.o_tf.shape[2]
        #
        hidden = 100
        layers = 3
        #
        with tf.variable_scope('ModelRNN'):
            input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])
            self.output = nn(input, [hidden] * layers + [dimo])


#
#
# class LSTM_InitStateSeqLen:
#     @store_args
#     def __init__(self, inputs_tf):
#         """The actor-critic network and related training code.
#
#         Args:
#             inputs_tf (dict of tensors): all necessary inputs for the network: the
#                 observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
#         """
#         print("Initializing model")
#         self.o_tf = inputs_tf['o']
#         self.u_tf = inputs_tf['u']
#         #
#         dimo = self.o_tf.shape[2]
#         #
#         rnn_size = 100
#         batch_size = 40
#         seq_length = 50
#         sequence_lengths = np.zeros(batch_size)
#         sequence_lengths += seq_length
#
#
#         cell_fn = LSTMCell
#
#         cell = cell_fn(rnn_size)
#
#         self.initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
#
#         with tf.variable_scope('ModelRNN') as mrnnscope:
#             input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])
#             output, last_state = tf.nn.dynamic_rnn(cell, input, sequence_length=sequence_lengths, initial_state=self.initial_state, swap_memory=True, dtype=tf.float32)
#             self.output = tf.layers.dense(output, dimo)
#
# class LSTM_SimpleNorm:
#     @store_args
#     def __init__(self, inputs_tf):
#         """The actor-critic network and related training code.
#
#         Args:
#             inputs_tf (dict of tensors): all necessary inputs for the network: the
#                 observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
#         """
#         print("Initializing model")
#         self.o_tf = inputs_tf['o']
#         self.u_tf = inputs_tf['u']
#         #
#         dimo = self.o_tf.shape[2]
#         #
#         rnn_size = 100
#         sequence_lengths = 50
#         # batch_size = 50
#
#         cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
#
#         use_recurrent_dropout = False
#         recurrent_dropout_prob = 0.5
#         # use_input_dropout = False
#         # use_output_dropout = False
#         is_training = True
#         use_layer_norm = True
#
#         if use_recurrent_dropout:
#             cell = cell_fn(rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=recurrent_dropout_prob)
#         else:
#             cell = cell_fn(rnn_size, layer_norm=use_layer_norm)
#
#         # self.initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
#
#         with tf.variable_scope('ModelRNN') as mrnnscope:
#             input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])
#             output, last_state = tf.nn.dynamic_rnn(cell, input, swap_memory=True, dtype=tf.float32)
#             self.output = tf.layers.dense(output, dimo)
#
#
# class LSTM_Simple2:
#     @store_args
#     def __init__(self, inputs_tf):
#         """The actor-critic network and related training code.
#
#         Args:
#             inputs_tf (dict of tensors): all necessary inputs for the network: the
#                 observation (o), the action (u) and the successive observation (o2) (o2 not required for the network)
#         """
#         print("Initializing model")
#         self.o_tf = inputs_tf['o']
#         self.u_tf = inputs_tf['u']
#         #
#         dimo = self.o_tf.shape[2]
#         #
#         size = 100
#         # layers = 3
#         #
#         with tf.variable_scope('ModelRNN'):
#             input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])
#             with tf.variable_scope('l_1'):
#                 out, states = tf.nn.dynamic_rnn(
#                     LSTMCell(size),
#                     input,
#                     dtype=tf.float32)
#             with tf.variable_scope('l_2'):
#                 out, states = tf.nn.dynamic_rnn(
#                     LSTMCell(size),
#                     out,
#                     dtype=tf.float32)
#
