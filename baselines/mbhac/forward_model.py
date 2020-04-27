import numpy as np
import tensorflow as tf
from baselines.mbhac.utils import layer


class ForwardModel():

    def __init__(self, sess, env, layer_number, mb_params, err_list_size):

        self.sess = sess
        self.model_name = 'model_' + str(layer_number)

        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        self.hidden_sizes = [int(size) for size in mb_params['hidden_size'].split(',')]
        self.eta = mb_params['eta']
        self.state_dim = env.state_dim
        self.learning_rate = mb_params['lr']

        self.err_list_size = err_list_size
        self.err_list = []

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

        hidden_layers = []
        for idx, layer_size in enumerate(self.hidden_sizes, start=1):
            with tf.variable_scope(name + 'fc_{}'.format(idx)):
                if idx <= 1:
                    hidden_layers.append(layer(action_state_ph, layer_size))
                else:
                    hidden_layers.append(layer(hidden_layers[-1], layer_size))

        with tf.variable_scope(name + 'fc_4'):
            pred = layer(hidden_layers[-1], self.state_dim, is_output=True)
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

    def normalize_bonus(self, bonus_lst):
        """ Bonus range between -1.0 and 0.0 """
        norm_bonus = (bonus_lst - self.min_err) / (self.max_err - self.min_err)
        return norm_bonus - 1.0

    def pred_bonus(self, action, state, s_next):
        s_next_prediction = self.pred_state(action, state)
        errs = (s_next_prediction - s_next) ** 2
        err = errs.mean(axis=1)

        if len(self.err_list) < self.err_list_size and err.size:
            self.err_list += err.tolist()
            # update bounds for normalization
            self.min_err = np.min(self.err_list)
            self.max_err = np.max(self.err_list)

        return self.normalize_bonus(err)

    def update(self, states, actions, new_states):
        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.action_ph: actions,
                self.state_ph: states,
                self.y: new_states
            })
        return loss
