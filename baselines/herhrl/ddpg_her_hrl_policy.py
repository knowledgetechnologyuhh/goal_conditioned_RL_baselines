from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.template.util import logger as log_formater
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.herhrl.normalizer import Normalizer
from baselines.herhrl.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam
from baselines.template.policy import Policy
from baselines.herhrl.hrl_policy import HRL_Policy
from baselines.herhrl.ddpg_her_hrl_policy_old import DDPG_HER_HRL_POLICY
def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

class DDPG_HER_HRL_POLICY(HRL_Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        self.ep_ctr = 0
        self.hist_bins = 50
        self.draw_hist_freq = 3
        self._reset_hists()
        self.shared_pi_err_coeff = kwargs['shared_pi_err_coeff']

        HRL_Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        self.hidden = hidden
        self.layers = layers
        self.max_u = max_u
        self.network_class = network_class
        self.sample_transitions = sample_transitions
        self.scope = scope
        self.subtract_goals = subtract_goals
        self.relative_goals = relative_goals
        self.clip_obs = clip_obs
        self.Q_lr = Q_lr
        self.pi_lr = pi_lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.clip_pos_returns = clip_pos_returns
        self.gamma = gamma
        self.polyak = polyak
        self.clip_return = clip_return
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.action_l2 = action_l2
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        self.stage_shapes['gamma'] = (None,)
        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *self.input_shapes[key])
                         for key, val in self.input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)
        buffer_shapes['p'] = (buffer_shapes['g'][0], 1)
        buffer_shapes['steps'] = buffer_shapes['p']
        buffer_size = self.buffer_size #// self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

        self.preproc_lr = (self.Q_lr + self.pi_lr) / 2

    def _reset_hists(self):
        self.hists = {"attn": None, "prob_in": None, "rnd": None}

    def draw_hists(self, img_dir):
        for hist_name, hist in self.hists.items():
            if hist is None:
                continue
            step_size = 1.0 / self.hist_bins
            xs = np.arange(0, 1, step_size)
            hist /= (self.ep_ctr * self.T)
            fig, ax = plt.subplots()
            ax.bar(xs, hist, step_size)
            plt.savefig(img_dir + "/{}_hist_l_{}_ep_{}.png".format(hist_name, self.h_level, self.ep_ctr))
        self._reset_hists()
        if self.child_policy is not None:
            self.child_policy.draw_hists(img_dir)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False, exploit=True, **kwargs):
        # noise_eps = noise_eps if not exploit else 0.
        # random_eps = random_eps if not exploit else 0.

        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf, policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        q = ret[1]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        noisy_u = u + noise
        u = np.clip(noisy_u, -self.max_u, self.max_u)
        random_u = np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - noisy_u)  # eps-greedy
        u += random_u
        u = u[0].copy()
        self.update_hists(feed)
        return u, q

    def update_hists(self, feed):
        vals = []
        hist_names_to_consider = []
        for hist_name, hist in self.hists.items():
            if hist_name in self.main.__dict__:
                hist_names_to_consider.append(hist_name)
                vals.append(eval("policy.{}".format(hist_name)))

        ret = self.sess.run(vals, feed_dict=feed)
        for val_idx, hist_name in enumerate(hist_names_to_consider):
            this_vals = ret[val_idx]
            this_hists = np.histogram(this_vals, self.hist_bins, range=(0, 1))
            if self.hists[hist_name] is None:
                self.hists[hist_name] = this_hists[0] / this_vals.shape[1]
            else:
                self.hists[hist_name] += this_hists[0] / this_vals.shape[1]

    def scale_and_offset_action(self, u):
        scaled_u = u.copy()
        scaled_u *= self.subgoal_scale
        scaled_u += self.subgoal_offset
        return scaled_u

    def inverse_scale_and_offset_action(self, scaled_u):
        u = scaled_u.copy()
        u -= self.subgoal_offset
        u /= self.subgoal_scale
        return u

    def store_episode(self, episode_batch, update_stats=True):

        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        # print("Storing Episode h-level = {}".format(self.h_level))
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # add transitions to normalizer

            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            # num_normalizing_transitions = episode_batch['u'].shape[1]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.o_stats.recompute_stats()

            self.g_stats.update(transitions['g'])
            self.g_stats.recompute_stats()

        self.ep_ctr += 1
        # print("Done storing Episode")

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()
        self.shared_preproc_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, preproc_loss, Q_grad, pi_grad, preproc_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.shared_preproc_loss_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,
            self.shared_preproc_grad_tf
        ])
        return critic_loss, actor_loss, preproc_loss, Q_grad, pi_grad, preproc_grad

    def _update(self, Q_grad, pi_grad, preproc_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        self.shared_preproc_adam.update(preproc_grad, self.preproc_lr)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, preproc_loss, Q_grad, pi_grad, preproc_grad = self._grads()
        self._update(Q_grad, pi_grad, preproc_grad)
        return critic_loss, actor_loss, preproc_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        # assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG_HRL agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        # target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        target_tf = tf.clip_by_value(batch_tf['r'] + tf.transpose(self.gamma * batch_tf['gamma']) * target_Q_pi_tf,
                                     *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        self.shared_q_err_coeff = 1.0 - self.shared_pi_err_coeff
        self.shared_preproc_loss_tf = (
                    self.shared_q_err_coeff * self.Q_loss_tf + self.shared_pi_err_coeff * self.pi_loss_tf)
        if "shared_preproc_err" in self.main.__dict__:
            self.shared_preproc_loss_tf += self.main.shared_preproc_err
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        shared_preproc_grads_tf = tf.gradients(self.shared_preproc_loss_tf, self._vars('main/shared_preproc'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        assert len(self._vars('main/shared_preproc')) == len(shared_preproc_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.shared_preproc_grads_vars_tf = zip(shared_preproc_grads_tf, self._vars('main/shared_preproc'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        self.shared_preproc_grad_tf = flatten_grads(grads=shared_preproc_grads_tf,
                                                    var_list=self._vars('main/shared_preproc'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)
        self.shared_preproc_adam = MpiAdam(self._vars('main/shared_preproc'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi') + self._vars('main/shared_preproc')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi') + self._vars('target/shared_preproc')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix='policy'):
        logs = []
        logs += [('buffer_size', int(self.buffer.current_size))]
        logs = log_formater(logs, prefix + "_{}".format(self.h_level))
        if self.child_policy is not None:
            child_logs = self.child_policy.logs(prefix=prefix)
            logs += child_logs

        return logs


    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)



