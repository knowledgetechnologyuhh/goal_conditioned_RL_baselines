import tensorflow as tf
from baselines.util import store_args, nn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
import numpy as np

class ModelRNN:
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
        self.o2_tf = inputs_tf['o2']
        self.loss_tf = inputs_tf['loss']

        self.max_batch_size = kwargs['model_train_batch_size']

        dimo = self.o_tf.shape[2]
        diml = self.loss_tf.shape[2]
        sizes = [100, 100]
        with tf.variable_scope('ModelRNN'):
            # create a BasicRNNCell
            self.rnn_cell = MultiRNNCell([tf.nn.rnn_cell.GRUCell(size) for size in sizes], state_is_tuple=False)

            # defining initial zero state
            self.initial_state = self.rnn_cell.zero_state(self.max_batch_size, dtype=tf.float32)

            input = tf.concat(axis=2, values=[self.o_tf, self.u_tf])

            rnn_out, rnn_state = tf.nn.dynamic_rnn(
                self.rnn_cell,
                input,
                initial_state=self.initial_state,
                dtype=tf.float32)

            self.rnn_state = rnn_state
            self.rnn_out = rnn_out
            self.output = tf.layers.dense(rnn_out, dimo)

        with tf.variable_scope('LossPredNN'):
            hidden = 100
            layers = 3
            loss_pred_in = tf.concat(axis=2, values=[self.o_tf, self.u_tf, rnn_out])
            loss_loss = nn(loss_pred_in, [hidden] * layers + [diml])
            self.loss_prediction_tf = tf.abs(loss_loss)

        self.obs_loss_per_step_tf = tf.reduce_mean(tf.abs(self.output - self.o2_tf), axis=2)
        self.loss_loss_per_step_tf = tf.reduce_mean(tf.abs(self.loss_prediction_tf - self.loss_tf), axis=2)

        self.obs_loss_tf = tf.reduce_mean(self.obs_loss_per_step_tf)
        self.loss_loss_tf = tf.reduce_mean(self.loss_loss_per_step_tf)
