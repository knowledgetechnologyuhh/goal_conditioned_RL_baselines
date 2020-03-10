from baselines.util import (store_args)
from baselines.template.policy import Policy
import numpy as np
from baselines.hac.layer import Layer
import tensorflow as tf
from baselines.hac.utils import EnvWrapper
from baselines import logger
import time

class HACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden_size, layers, polyak, batch_size, Q_lr, pi_lr, norm_eps, norm_clip, max_u,
            action_l2, clip_obs, scope, T, rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
            sample_transitions, gamma,time_scale, subgoal_test_perc, n_layers, model_based, mb_hidden_size, mb_lr, eta,reuse=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)
        #  print(input_dims, buffer_size, hidden, layers, polyak, batch_size, Q_lr, pi_lr, norm_eps, norm_clip, max_u,
        #      action_l2, clip_obs, scope, T, rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
        #      sample_transitions, gamma, reuse,time_scale, subgoal_test_perc, kwargs)

        self.verbose = False
        self.Q_values = True
        self.n_layers = n_layers

        # TODO: Why we get tuples?
        time_scale = time_scale[0]
        subgoal_test_perc = subgoal_test_perc[0]
        self.buffer_size = buffer_size[0]
        self.batch_size = batch_size[0]
        self.model_based = model_based

        self.env = EnvWrapper(kwargs['make_env']().env, n_layers, time_scale, input_dims, max_u, self)
        agent_params = {
            "subgoal_test_perc": subgoal_test_perc,
            "subgoal_penalty": -time_scale,
            "atomic_noise": [0.1 for i in range(input_dims['u'])],
            "subgoal_noise": [0.1 for i in range(len(self.env.sub_goal_thresholds))],
            "n_layers": n_layers,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "time_scale": time_scale,
            "hidden_size": hidden_size,
            "Q_lr": Q_lr,
            "pi_lr": pi_lr,
            "model_based": self.model_based,
            "mb_params": {
                "hidden_size": mb_hidden_size,
                "lr": mb_lr,
                "eta": eta,
                }
        }

        with tf.variable_scope(self.scope):
            self._create_networks(agent_params)

        # goal_array stores goal for each layer of agent.
        self.goal_array = [None for i in range(n_layers)]
        self.current_state = None
        self.steps_taken = 0
        self.total_steps = 0

    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self,env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.n_layers)]
        max_lay_achieved = None
        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env._obs2subgoal(self.current_state)
        proj_end_goal = env._obs2goal(self.current_state)

        for i in range(self.n_layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.n_layers - 1:

                # Check dimensions are appropriate
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.sub_goal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.sub_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def set_train_mode(self):
        self.test_mode = False

    def set_test_mode(self):
        self.test_mode = True

    def _create_networks(self, agent_params):
        logger.info("Creating a HAC agent with action space %d x %s..." % (self.dimu, self.max_u))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]
        # Create agent with number of levels specified by user
        self.layers = [Layer(i,self.env,self.sess, agent_params) for i in range(self.n_layers)]
        self.sess.run(tf.global_variables_initializer())

    # Update actor and critic networks for each layer
    def learn(self, num_updates):
        learn_summaries = []

        for i in range(len(self.layers)):
            learn_summay = self.layers[i].learn(num_updates)
            learn_summaries.append(learn_summay)

        return learn_summaries


    # Train agent for an episode
    def train(self,env, episode_num, eval_data, num_updates):
        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.n_layers - 1] = env._sample_goal()
        env.display_end_goal(self.goal_array[self.n_layers - 1])

        if self.verbose:
            print("Next End Goal: ", self.goal_array[self.n_layers - 1])

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env._reset_sim(self.goal_array[self.n_layers - 1])['observation']

        if self.verbose:
            print("Initial State: ", self.current_state[:3])

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, eval_data, max_lay_achieved = self.layers[self.n_layers-1].\
            train(self, env, episode_num=episode_num, eval_data=eval_data)

        train_duration = 0
        # Update actor/critic networks if not testing
        if not self.test_mode:
            train_start = time.time()
            learn_summaries = self.learn(num_updates)

            for l in range(self.n_layers):
                learn_summary = learn_summaries[l]
                for k,v in learn_summary.items():
                    eval_data["train_{}/{}".format(l,k)] = v

            train_duration += time.time() - train_start

        self.total_steps += self.steps_taken
        # Return whether end goal was achieved
        return goal_status[self.n_layers-1], eval_data, train_duration


    def logs(self, prefix=''):
        logs = []
        logs += [('steps', self.total_steps)]

        if prefix != '' and not prefix.endswith('/'):
            logs = [(prefix + '/' + key, val) for key, val in logs]

        return logs

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def __getstate__(self):
        #  TODO: modfiy exclude array #
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic',
                             # TODO: fix layers
                             'obs2preds_buffer', 'obs2preds_model', 'eval_data', 'layers']
        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name and 'obs2preds_buffer' not in x.name])
        return state


