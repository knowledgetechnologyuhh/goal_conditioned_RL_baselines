from baselines.util import (store_args)
from baselines.template.policy import Policy
import numpy as np
from baselines.chac.layer import Layer
from baselines.chac.utils import prepare_env
from baselines import logger
import torch
import time


class CHACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, T, rollout_batch_size, agent_params, env, verbose=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            logger.info('Use GPU:{} {}', torch.cuda.current_device(),
                  torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            logger.info('Use CPU')

        self.verbose = verbose
        self.n_levels = agent_params['n_levels']
        self.env = env
        self.fw = agent_params['fw']
        self._create_networks(agent_params)

        # goal_array stores goal for each layer of agent.
        self.goal_array = [None] * self.n_levels
        self.current_state = None
        self.steps_taken = 0
        self.total_steps = 0

    def check_goals(self, env):
        """Determine whether or not each layer's goal was achieved. Also, if applicable, return the highest level whose goal was achieved."""
        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False] * self.n_levels
        max_lay_achieved = None

        # Project current state onto relevant goal spaces
        proj_end_goal = env.project_state_to_end_goal(self.current_state)
        proj_subgoal = env.project_state_to_sub_goal(self.current_state)

        for i in range(self.n_levels):
            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.n_levels - 1:
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(env.end_goal_thresholds), \
                        "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"
                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.sub_goal_thresholds), \
                        "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"
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
        logger.info("Creating a CHAC agent")
        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]
        # Create agent with number of levels specified by user
        self.layers = [
            Layer(i, self.env, agent_params, self.device)
            for i in range(self.n_levels)
        ]

    def learn(self, num_updates):
        """Update actor and critic networks for each layer"""
        return [
            self.layers[i].learn(num_updates) for i in range(len(self.layers))
        ]

    def train(self, env, episode_num, eval_data, num_updates):
        """Train agent for an episode"""
        obs = env.reset()
        self.current_state = obs['observation']

        if self.verbose:
            print("Initial State: ", self.current_state[:3])

        self.goal_array[self.n_levels - 1] = obs['desired_goal']
        env.wrapped_env.final_goal = obs['desired_goal']

        if self.verbose:
            print("Next End Goal: ", env.wrapped_env.final_goal, env.wrapped_env.goal)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, eval_data, max_lay_achieved = self.layers[self.n_levels-1].\
            train(self, env, episode_num=episode_num, eval_data=eval_data)

        train_duration = 0
        # Update actor/critic networks if not testing
        if not self.test_mode:
            train_start = time.time()
            learn_summaries = self.learn(num_updates)

            for l in range(self.n_levels):
                learn_summary = learn_summaries[l]
                for k, v in learn_summary.items():
                    eval_data["train_{}/{}".format(l, k)] = v

            train_duration += time.time() - train_start

        self.total_steps += self.steps_taken
        # Return whether end goal was achieved
        return goal_status[self.n_levels - 1], eval_data, train_duration

    def logs(self, prefix=''):
        logs = []
        logs += [('steps', self.total_steps)]

        if prefix != '' and not prefix.endswith('/'):
            logs = [(prefix + '/' + key, val) for key, val in logs]

        return logs

    def __getstate__(self):
        excluded_subnames = [
            '_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats', 'main',
            'target', 'lock', 'env', 'sample_transitions', 'stage_shapes',
            'create_actor_critic', 'obs2preds_buffer', 'obs2preds_model',
            'eval_data', 'layers', 'goal_array', 'total_steps',
            'current_state', 'device'
        ]

        state = {
            k: v
            for k, v in self.__dict__.items()
            if all([not subname in k for subname in excluded_subnames])
        }
        state['torch'] = {}

        # save pytoch model weights
        for layer in self.layers:
            l = str(layer.level)
            state['torch']['actor' + l] = layer.actor.cpu().state_dict()
            state['torch']['critic' + l] = layer.critic.cpu().state_dict()
            if hasattr(layer, 'state_predictor'):
                state['torch']['fw_model' + l] = layer.state_predictor.cpu().state_dict()
                state['fw_model' + l + 'err_list'] = layer.state_predictor.err_list

            # move back, just in case
            layer.actor.to(self.device)
            layer.critic.to(self.device)
            if hasattr(layer, 'state_predictor'):
                layer.state_predictor.to(self.device)

        return state

    def __setstate__(self, state):
        agent_params = state['agent_params']
        env_name = state['info']['env_name']
        state['n_levels'] = agent_params['n_levels']
        state['env'] = prepare_env(env_name, agent_params['n_levels'],
                                   agent_params['time_scale'],
                                   state['input_dims'])
        self.__init__(**state)
        self.env.agent = self

        # load network states
        for layer in self.layers:
            l = str(layer.level)
            layer.actor.load_state_dict(state['torch']['actor' + l])
            layer.critic.load_state_dict(state['torch']['critic' + l])
            if hasattr(layer, 'state_predictor'):
                layer.state_predictor.load_state_dict(state['torch']['fw_model' + l])
                layer.state_predictor.err_list = state['fw_model' + l + 'err_list']
                layer.state_predictor.min_err = np.min(state['fw_model' + l + 'err_list'])
                layer.state_predictor.max_err = np.max(state['fw_model' + l + 'err_list'])
