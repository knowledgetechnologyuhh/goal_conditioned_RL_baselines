import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
import collections
import numpy as np
from baselines import logger
from baselines.template.policy import Policy
from baselines.common.mpi_adam import MpiAdam

from collections import OrderedDict
# from baselines.model_based.replay_buffer import ReplayBuffer
from baselines.model_based.model_replay_buffer import ModelReplayBuffer
from baselines.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class MBPolicy(Policy):
    @store_args
    def __init__(self, input_dims, model_buffer_size, model_network_class, scope, T, rollout_batch_size, model_lr, model_train_batch_size, **kwargs):

        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
        # self.buffer_size=10
        self.env = kwargs['env']
        self.scope = scope
        self.create_model = import_function(model_network_class)
        self.model_loss_history = collections.deque(maxlen=100)
        self.model_loss_exploration_threshold = 0.00 # The average model loss threshold over the last 100 episodes to consider the model realistic and to start model-based planning
        #
        # # Create network.
        model_stage_shapes = OrderedDict()
        time_dim = self.T
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            if key in ['o']:
                model_stage_shapes[key] = (None, time_dim, *self.input_shapes[key])
                model_stage_shapes[key +"2"] = (None, time_dim, *self.input_shapes[key])
            if key in ['u']:
                model_stage_shapes[key] = (None, time_dim, *self.input_shapes[key])

        self.model_stage_shapes = model_stage_shapes

        with tf.variable_scope(self.scope):
            self.model_staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.model_stage_shapes.keys()],
                shapes=[(None, None, shape[2]) for _, shape in self.model_stage_shapes.items()])

            self.model_buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=(None, None, shape[2])) for _, shape in self.model_stage_shapes.items()]
            self.model_stage_op = self.model_staging_tf.put(self.model_buffer_ph_tf)

            self._create_network(reuse=False)


        # Initialize the model replay buffer.
        self.model_replay_buffer = ModelReplayBuffer(model_stage_shapes, model_buffer_size)
        # pass
        print("done init MBPolicy")

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def get_init_zero_state(self):
        pred_net2 = self.create_model(
            {'o': tf.placeholder(tf.float32, shape=(None, None, 1)), 'o2': tf.placeholder(tf.float32, shape=(None, None, 1)), 'u': tf.placeholder(tf.float32, shape=(None, None, 1))},
            **self.__dict__)
        if 'initial_state' in pred_net2.__dict__.keys():
            # initial_state = tuple([int(d) for d in pred_net2.init_state.shape])
            return pred_net2.initial_state
        # if pred_net2.__dict__
        else:
            return None

    def _create_network(self, reuse=False):
        logger.info("Creating a model-based agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()
        # # mini-batch sampling.
        model_batch = self.model_staging_tf.get()
        model_batch_tf = OrderedDict([(key, model_batch[i])
                                for i, key in enumerate(self.model_stage_shapes.keys())])
        # # networks
        with tf.variable_scope('model') as ms:
            self.prediction_model = self.create_model(model_batch_tf, **self.__dict__)
            ms.reuse_variables()

        # self.obs_loss_tf = tf.reduce_mean(tf.square(self.prediction_model.output - model_batch_tf['o2']))
        model_grads = tf.gradients(self.prediction_model.obs_loss_tf, self._vars('model/ModelRNN'))
        self.model_grads_tf = flatten_grads(grads=model_grads, var_list=self._vars('model/ModelRNN'))

        # # optimizers
        self.pred_adam = MpiAdam(self._vars('model/ModelRNN'), scale_grad_by_procs=False)

        # # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()

    def get_actions(self, o, ag, g, policy_action_params=None):
        return self.get_actions_explore(o, ag, g)

    def get_actions_explore(self, o, ag, g, policy_action_params=None):
        u_s = []
        for ro_idx in range(self.rollout_batch_size):
            u = np.random.randn(self.dimu)
            u_s.append(u)
        u_s = np.array(u_s)
        if len(u_s) == 1:
            return u_s[0]
        else:
            return u_s

    def get_actions_policy(self, o, ag, g, policy_action_params=None):
        batch = self.model_replay_buffer.sample(self.model_replay_buffer.current_size)
        ep_len = batch['o2'].shape[1]
        ep_steps_remaining = ep_len - self.env.step_ctr
        u_s = []
        for ro_idx in range(self.rollout_batch_size):
            obs_goal = self.env._obs2goal(o[ro_idx])
            if self.env._is_success(g[ro_idx], obs_goal):
                u = np.zeros(self.dimu)

            elif np.mean(self.model_loss_history) < self.model_loss_exploration_threshold:
                # print("Using model and experience to find appropriate action towards achieving the goal")

                smallest_dist_to_goal = np.finfo(np.float32).max
                smallest_dist_to_goal_obs_idx = -1
                smallest_dist_to_goal_obs_seq_idx = -1
                smallest_dist_to_obs = np.finfo(np.float32).max
                smallest_dist_to_obs_obs_idx = -1
                smallest_dist_to_obs_obs_seq_idx = -1

                # Find sequence with state that is closest to goal and sequence with state that is closest to observation.
                for obs_seq_idx, obs_seq in enumerate(batch['o2']):
                    for obs_idx, obs in enumerate(obs_seq):
                        obs_goal = self.env._obs2goal(obs)
                        goal_dist = np.linalg.norm(obs_goal - g[ro_idx], axis=-1)
                        if goal_dist < smallest_dist_to_goal:
                            smallest_dist_to_goal = goal_dist
                            smallest_dist_to_goal_obs_idx = obs_idx
                            smallest_dist_to_goal_obs_seq_idx = obs_seq_idx
                        if (obs_idx + 1) < ep_len:
                            obs_dist = np.linalg.norm(obs - o[ro_idx], axis=-1)
                            if obs_dist < smallest_dist_to_obs:
                                smallest_dist_to_obs = obs_dist
                                smallest_dist_to_obs_obs_idx = obs_idx + 1
                                smallest_dist_to_obs_obs_seq_idx = obs_seq_idx

                # Now connect both sequences at minimal intersection point, such that the resulting sequence contains at most ep_steps_remaining transitions
                smallest_obs_dist = np.finfo(np.float32).max
                intersection_idxs = (smallest_dist_to_obs_obs_idx, smallest_dist_to_goal_obs_idx)
                for og_idx, obs_goal in enumerate(batch['o2'][smallest_dist_to_goal_obs_seq_idx][:smallest_dist_to_goal_obs_idx]):
                    inter_to_goal_steps = smallest_dist_to_goal_obs_idx - og_idx
                    for oo_idx, obs_obs in enumerate(batch['o'][smallest_dist_to_obs_obs_seq_idx][smallest_dist_to_obs_obs_idx:]):
                        obs_to_inter_steps = oo_idx
                        total_steps = inter_to_goal_steps + obs_to_inter_steps
                        if total_steps > ep_steps_remaining:
                            continue
                        this_dist = np.linalg.norm(obs_goal - obs_obs, axis=-1)
                        if this_dist < smallest_obs_dist:
                            intersection_idxs = (oo_idx + smallest_dist_to_obs_obs_idx, og_idx)
                            smallest_obs_dist = this_dist

                actions = np.concatenate(
                    (batch['u'][smallest_dist_to_obs_obs_seq_idx][smallest_dist_to_obs_obs_idx:intersection_idxs[0]],
                     batch['u'][smallest_dist_to_goal_obs_seq_idx][intersection_idxs[1]:smallest_dist_to_goal_obs_idx]))

                success = False
                next_o = o[ro_idx]
                next_s = None # The optional internal state in case of using RNNs.
                for u in actions:
                    next_o, next_s = self.forward_step(u,next_o, next_s)
                    next_o_goal = self.env._obs2goal(next_o)
                    if self.env._is_success(g[ro_idx], next_o_goal):
                        u = actions[0]
                        # print("I found a plan that should work, according to the learned forward model.")
                        success = True
                        break
                if success is False:
                    u = np.random.randn(self.dimu)
            else:
                # print("Selecting action that maximizes surprisal-based exploration")
                u = np.random.randn(self.dimu)
            u_s.append(u)
        u_s = np.array(u_s)
        if len(u_s) == 1:
            return u_s[0]
        else:
            return u_s

    def store_episode(self, episode_batch, update_stats=True):
        # print("Storing episode batch")
        episodes = []
        for e_idx in range(len(episode_batch['o'])):
            episode = {}
            episode['o'] = episode_batch['o'][e_idx][:-1]
            episode['u'] = episode_batch['u'][e_idx]
            episode['o2'] = episode_batch['o'][e_idx][1:]
            episodes.append(episode)
        self.model_replay_buffer.store_episode(episodes)
        pass


    def sample_batch(self, batch_size=None):
        # print("Sampling batch")
        if batch_size is None:
            batch_size = self.model_train_batch_size
        batch_dict = self.model_replay_buffer.sample(batch_size)
        batch = [batch_dict[key] for key in self.model_stage_shapes.keys()]
        return batch

    def forward_step(self,u,o,s):

        bs = self.model_train_batch_size

        single_step = [np.array([[o]] * bs), np.array([[o]] * bs), np.array([[u]] * bs)]
        # batch = self.sample_batch(bs)
        # single_step = batch

        fd = {self.prediction_model.o_tf: single_step[0], self.prediction_model.u_tf: single_step[2]}
        if s is not None and 'initial_state' in self.prediction_model.__dict__:
            fd[self.prediction_model.initial_state] = s

        fetches = [self.prediction_model.output]
        if 'state' in self.prediction_model.__dict__:
            fetches.append(self.prediction_model.state)
            o2, s2 = self.sess.run(fetches, feed_dict=fd)
        else:
            o2 = np.array(self.sess.run(fetches, feed_dict=fd)[0])
            s2 = None

        # fd = dict(zip(self.model_buffer_ph_tf, single_step))
        # self.sess.run(self.model_stage_op, feed_dict=fd)
        # o2, s2 = self.sess.run([
        #     self.prediction_model.output,
        #     self.prediction_model.state
        # ])

        next_o = np.array(o2[0][0])
        return next_o, s2

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.model_buffer_ph_tf) == len(batch)
        fd = dict(zip(self.model_buffer_ph_tf, batch))
        self.sess.run(self.model_stage_op, feed_dict=fd)

    def _update(self, model_grads):
        self.pred_adam.update(model_grads, self.model_lr)

    def _grads(self):
        # Avoid feed_dict here for performance!
        model_loss, model_grads = self.sess.run([
            self.obs_loss_tf,
            self.model_grads_tf
        ])
        return model_loss, model_grads


    def train(self):
        # print("Training")
        batch = self.sample_batch()
        fetches = [
            self.prediction_model.obs_loss_tf,
            self.model_grads_tf
        ]
        fd = {self.prediction_model.o_tf : batch[0],
              self.prediction_model.o2_tf : batch[1],
              self.prediction_model.u_tf : batch[2]}
        model_loss, model_grads = self.sess.run(fetches, fd)


        # self.stage_batch()
        # model_loss, model_grads = self._grads()
        self._update(model_grads)
        self.model_loss_history.append(model_loss)
        return model_loss

    def logs(self, prefix=''):
        logs = []
        # logs += [('stats/some_stat_value', 0)]
        # logs = []
        # logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        # logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        # logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        # logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _sync_optimizers(self):
        self.pred_adam.sync()


    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        # [print(key, ": ", item) for key, item in self.__dict__.items()]
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'model_replay_buffer', 'sess', '_stats',
                             'prediction_model', 'lock', 'env',
                             'stage_shapes', 'model_stage_shapes', 'create_model']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['model_buffer_size'] = self.model_buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'model_replay_buffer' not in x.name])
        return state

    def __setstate__(self, state):
        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'model_replay_buffer' not in x.name]
        assert (len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)


