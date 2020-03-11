import numpy as np
import tensorflow as tf
from baselines.hac.utils import layer


class ForwardModel():

    def __init__(self, sess, env, layer_number, mb_params):

        self.sess = sess
        self.model_name = 'model_' + str(layer_number)

        if layer_number == 0:
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset

        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.h = [mb_params['hidden_size']] * 3
        self.eta = mb_params['eta']
        self.state_dim = env.state_dim
        self.learning_rate = mb_params['lr']

        self.action_ph, self.state_ph, self.y, self.pred, self.loss, self.optimizer \
            = self._build_graph(layer_number)

    def _build_graph(self, layer_number):
        name = self.model_name

        with tf.variable_scope(name + 'action_ph'):
            action_ph = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
        with tf.variable_scope(name + 'state_ph'):
            state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        with tf.variable_scope(name + 'action_state_ph'):
            action_state_ph = tf.concat([action_ph, state_ph], axis=1)
        with tf.variable_scope(name + 'target_state_ph'):
            y = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        with tf.variable_scope(name + 'fc_1'):
            fc1 = layer(action_state_ph, self.h[0])
        with tf.variable_scope(name + 'fc_2'):
            fc2 = layer(fc1, self.h[1])
        with tf.variable_scope(name + 'fc_3'):
            fc3 = layer(fc2, self.h[2])
        with tf.variable_scope(name + 'fc_4'):
            pred = layer(fc3, self.state_dim, is_output=True)
        with tf.variable_scope(name + 'loss'):
            loss = tf.losses.mean_squared_error(y, pred)
        with tf.variable_scope(name + 'optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return action_ph, state_ph, y, pred, loss, optimizer

    def pred_state(self, action, state):
        assert len(action[0]) == self.action_space_size
        assert len(state[0]) == self.state_dim
        return self.sess.run(
            self.pred, feed_dict={
                self.action_ph: action,
                self.state_ph: state
            })

    def pred_bonus(self, action, state, s_next):
        s_next_prediction = self.pred_state(action, state)
        err = ((np.array(s_next_prediction) - np.array(s_next)) ** 2).mean()
        assert type(err) == np.float64
        scale_curiosity = self.eta/2
        bonus = scale_curiosity * err
        return bonus

    def update(self, states, actions, new_states):
        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.action_ph: actions,
                self.state_ph: states,
                self.y: new_states
            })
        return loss
