import numpy as np
import torch
from baselines.chac.experience_buffer import ExperienceBuffer
from baselines.chac.actor import Actor
from baselines.chac.critic import Critic
from baselines.chac.forward_model import ForwardModel

class Layer():
    def __init__(self, layer_number, env, agent_params, device):
        self.layer_number = layer_number
        self.n_layers = agent_params['n_layers']
        self.time_scale = agent_params['time_scale']
        self.subgoal_test_perc = agent_params['subgoal_test_perc']
        self.fw = agent_params['fw']
        self.device = device

        # Set time limit for each layer. If agent uses only 1 layer, time limit
        # is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if self.n_layers > 1:
            self.time_limit = self.time_scale
        else:
            self.time_limit = env.max_actions

        self.current_state = None
        self.goal = None
        #  env.wrapped_env.goal_hierarchy[self.layer_number] = self.goal

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params['buffer_size']

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit * self.subgoal_test_perc)

        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.n_layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)

        act_dim = goal_dim = env.subgoal_dim
        if self.layer_number == 0:
            # Actions of lowest layer are real actions in environment
            act_dim = env.action_dim
        if self.layer_number == self.n_layers - 1:
            # Goals of highest layer are real goals of environment
            goal_dim = env.end_goal_dim

        self.replay_buffer = ExperienceBuffer(self.buffer_size, agent_params['batch_size'], env.state_dim, act_dim, goal_dim)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        print('Layer:', self.layer_number)
        # Initialize networks
        self.actor = Actor(env, self.layer_number, self.n_layers,
                hidden_size=agent_params['hidden_size'], learning_rate=agent_params['pi_lr']).to(self.device)
        print(self.actor)

        self.critic = Critic(env, self.layer_number, self.n_layers, self.time_scale,
                hidden_size=agent_params['hidden_size'], learning_rate=agent_params['q_lr']).to(self.device)
        print(self.critic)

        if self.fw:
            self.state_predictor = ForwardModel(env, self.layer_number, agent_params['fw_params'], self.buffer_size).to(self.device)
            print(self.state_predictor)

        # Parameter determines degree of noise added to actions during training
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False
        self.subgoal_penalty = agent_params["subgoal_penalty"]
        self.curiosity_hist = []
        self.q_values = []


    # Add noise to provided action
    def add_noise(self,action, env):

        # Noise added will be percentage of range
        if self.layer_number == 0:
            action_bounds = env.action_bounds
            action_offset = env.action_offset
        else:
            action_bounds = env.subgoal_bounds_symmetric
            action_offset = env.subgoal_bounds_offset

        assert len(action) == len(action_bounds), "Action bounds must have same dimension as action"
        assert len(action) == len(self.noise_perc), "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        for i in range(len(action)):
            action[i] += np.random.normal(0,self.noise_perc[i] * action_bounds[i])
            action[i] = max(min(action[i], action_bounds[i]+action_offset[i]), -action_bounds[i]+action_offset[i])

        return action


    def get_random_action(self, env):

        if self.layer_number == 0:
            action = np.zeros((env.action_dim))
        else:
            action = np.zeros((env.subgoal_dim))

        # Each dimension of random action should take some value in the dimension's range
        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] = np.random.uniform(-env.action_bounds[i] + env.action_offset[i], env.action_bounds[i] + env.action_offset[i])
            else:
                action[i] = np.random.uniform(env.subgoal_bounds[i][0],env.subgoal_bounds[i][1])

        return action


    # Function selects action using an epsilon-greedy policy
    def choose_action(self, agent, env, subgoal_test):

        current_state_tensor = torch.FloatTensor(self.current_state).view(1, -1).to(self.device)
        goal_tensor = torch.FloatTensor(self.goal).view(1, -1).to(self.device)
        # If testing mode or testing subgoals, action is output of actor network without noise
        if agent.test_mode or subgoal_test:
            with torch.no_grad():
                action = self.actor(current_state_tensor, goal_tensor)[0].cpu().numpy()
            action_type = "Policy"
            next_subgoal_test = subgoal_test
        else:
            if np.random.random_sample() > 0.2:
                with torch.no_grad():
                    action = self.add_noise(self.actor(current_state_tensor, goal_tensor)[0].cpu().numpy(), env)
                action_type = "Noisy Policy"
            else:
                action = self.get_random_action(env)
                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < self.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False

        return action, action_type, next_subgoal_test

    def perform_action_replay(self, hindsight_action, next_state, goal_status):
        """Create action replay transition by evaluating hindsight action given original goal
           Determine reward (0 if goal achieved, -1 otherwise) and finished boolean """
        # The finished boolean is used for determining the target for Q-value updates
        finished = goal_status[self.layer_number]
        reward = 0.0 if finished else -1.0
        # Transition will take the form [old state, hindsight_action, reward, next_state, goal, terminate boolean, None]
        transition = np.array([self.current_state, hindsight_action, reward, next_state, self.goal, finished, None])
        self.replay_buffer.add(transition)


    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, env, total_layers):
        """Create initial goal replay transitions
        Create transition evaluating hindsight action for some goal to be determined in future.
        Goal will be ultimately be selected from states layer has traversed through.
        Transition will be in the form [old state, hindsight action, reward = None, next state,
            goal = None, finished = None, next state projeted to subgoal/end goal space]"""

        if self.layer_number == total_layers - 1:
            hindsight_goal = env.project_state_to_end_goal(next_state)
        else:
            hindsight_goal = env.project_state_to_sub_goal(next_state)

        transition = np.array([self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal])
        self.temp_goal_replay_storage.append(transition)

    # Return reward given provided goal and goal achieved in hindsight
    def get_reward(self,new_goal, hindsight_goal, goal_thresholds):
        assert len(new_goal) == len(hindsight_goal) == len(goal_thresholds),\
                "Goal, hindsight goal, and goal thresholds do not have same dimensions"
        achived = np.all(np.absolute(new_goal - hindsight_goal) < goal_thresholds, axis=-1)
        return achived - 1.0



    def finalize_goal_replay(self,goal_thresholds):
        """Finalize goal replay by filling in goal, reward, and finished boolean
        for the preliminary goal replay transitions created before"""
        # Choose transitions to serve as goals during goal replay.  The last transition will always be used
        num_trans = len(self.temp_goal_replay_storage)

        if num_trans == 0:
            return

        num_replay_goals = self.num_replay_goals

        # If fewer transitions that ordinary number of replay goals, lower number of replay goals
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        indices = np.zeros((num_replay_goals))
        indices[:num_replay_goals-1] = np.random.randint(num_trans,size=num_replay_goals-1)
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)

        # For each selected transition, update the goal dimension of the selected transition and all prior transitions
        # by using the next state of the selected transition as the new goal.  Given new goal, update the reward and
        # finished boolean as well.
        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)

            new_goal = trans_copy[int(indices[i])][6]
            for index in range(num_trans):
                # Update goal to new goal
                trans_copy[index][4] = new_goal

                # Update reward
                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds)

                # Update finished boolean based on reward
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False

                self.replay_buffer.add(trans_copy[index])

        # Clear storage for preliminary goal replay transitions at end of goal replay
        self.temp_goal_replay_storage = []


    def penalize_subgoal(self, subgoal, next_state, test_fail=True):
        """Create transition penalizing subgoal if necessary. The target Q-value when this transition is used will ignore
        next state as the finished boolena = True.  Change the finished boolean to False, if you would like the subgoal
        penalty to depend on the next state."""
        transition = np.array([self.current_state, subgoal, self.subgoal_penalty if test_fail else 0.0, next_state, self.goal, True, None])
        self.replay_buffer.add(transition)


    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):
        """
        Return to higher level if
        (i) a higher level goal has been reached,
        (ii) maxed out episode time steps (env.max_actions)
        (iii) not testing and layer is out of attempts, and
        (iv) testing, layer is not the highest level, and layer is out of attempts.
        -----------------------------------------------------------------------------------
        NOTE: during testing, highest level will continue to ouput subgoals until either
        (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.
        """

        assert env.step_ctr == agent.steps_taken, "Step counter of env and agent should be equal"
        # Return to previous level when any higher level goal achieved.
        # NOTE: if not testing and agent achieves end goal, training will continue until
        # out of time (i.e., out of time steps or highest level runs out of attempts).
        # This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif env.step_ctr >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.test_mode and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.test_mode and self.layer_number < agent.n_layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False

    def train(self, agent, env, subgoal_test=False, episode_num=None, eval_data={}):
        """Learn to achieve goals with actions belonging to appropriate time scale.
        "goal_array" contains the goal states for the current layer and all higher layers"""
        train_test_prefix = 'test_{}/'.format(self.layer_number) if agent.test_mode else 'train_{}/'.format(self.layer_number)
        if self.layer_number > 0:
            if "{}subgoal_succ".format(train_test_prefix) not in eval_data:
                eval_data["{}subgoal_succ".format(train_test_prefix)] = []
            if "{}n_subgoals".format(train_test_prefix) not in eval_data:
                eval_data["{}n_subgoals".format(train_test_prefix)] = 0

        if "{}Q".format(train_test_prefix) not in eval_data:
            eval_data["{}Q".format(train_test_prefix)] = []

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and agent.env.visualize and agent.n_layers > 1:
            env.display_subgoals(agent.goal_array)

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0

        while True:
            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test = self.choose_action(agent, env, subgoal_test)
            current_state_tensor = torch.FloatTensor(self.current_state).view(1, -1).to(self.device)
            goal_tensor =  torch.FloatTensor(self.goal).view(1, -1).to(self.device)
            action_tensor = torch.FloatTensor(action).view(1, -1).to(self.device)
            with torch.no_grad():
                q_val = self.critic(current_state_tensor, goal_tensor, action_tensor)
            eval_data["{}Q".format(train_test_prefix)] += [q_val[0].item()]
            self.q_values += [q_val[0].item()]

            # If next layer is not bottom level, propose subgoal for next layer to achieve and determine
            # whether that subgoal should be tested
            if self.layer_number > 0:
                agent.goal_array[self.layer_number - 1] = action

                goal_status, eval_data, max_lay_achieved = agent.layers[self.layer_number - 1].\
                    train(agent, env, next_subgoal_test, episode_num, eval_data)

                eval_data["{}subgoal_succ".format(train_test_prefix)] += [1.0 if goal_status[self.layer_number-1] else 0.0]
                eval_data["{}n_subgoals".format(train_test_prefix)] += 1

            # If layer is bottom level, execute low-level action
            else:
                next_state = env.execute_action(action)
                agent.steps_taken += 1

                if agent.verbose and env.step_ctr >= env.max_actions:
                    print("Out of actions (Steps: %d)" % env.step_ctr)

                agent.current_state = next_state

                # Determine whether any of the goals from any layer was achieved
                # and, if applicable, the highest layer whose goal was achieved
                goal_status, max_lay_achieved = agent.check_goals(env)

            attempts_made += 1

            # This is very slow, only use for testing and render=True
            if agent.env.graph and self.fw and agent.env.visualize and self.state_predictor.err_list:
                agent_state_tensor = torch.FloatTensor(self.current_state).view(1, -1).to(self.device)
                self.state_predictor.eval()
                curi = self.state_predictor.pred_bonus(action_tensor, current_state_tensor, agent_state_tensor)
                self.curiosity_hist += curi.tolist()

            # Print if goal from current layer has been achieved
            if agent.verbose and goal_status[self.layer_number]:

                if self.layer_number < agent.n_layers - 1:
                    print("SUBGOAL ACHIEVED")

                print("\nEpisode %d, Layer %d, Attempt %d Goal Achieved" % (episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)

                if self.layer_number == agent.n_layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_sub_goal(agent.current_state))

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = env.project_state_to_sub_goal(agent.current_state)

            # Next, create hindsight transitions if not testing
            if not agent.test_mode:
                # Create action replay transition by evaluating hindsight action given current goal
                self.perform_action_replay(hindsight_action, agent.current_state, goal_status)

                # Create preliminary goal replay transitions.  The goal and reward in these transitions will be
                # finalized when this layer has run out of attempts or the goal has been achieved.
                self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.n_layers)
                #
                # # Penalize subgoals if subgoal testing and subgoal was missed by lower layers after maximum number of attempts
                test_fail = agent.layers[self.layer_number - 1].maxed_out
                if self.layer_number > 0 and next_subgoal_test and test_fail:
                    self.penalize_subgoal(action, agent.current_state, test_fail)

            # Print summary of transition
            if agent.verbose:

                print("\nEpisode %d, Level %d, Attempt %d" % (episode_num, self.layer_number,attempts_made))
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)

                if self.layer_number == agent.n_layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_sub_goal(agent.current_state))

                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)

            # Update state of current layer
            self.current_state = agent.current_state

            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or \
                    env.step_ctr >= env.max_actions or attempts_made >= self.time_limit:

                if agent.verbose and self.layer_number == agent.n_layers-1:
                    print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True

                # If not testing, finish goal replay by filling in missing goal and reward values before returning to
                # prior level.
                if not agent.test_mode:
                    if self.layer_number == agent.n_layers - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.sub_goal_thresholds

                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, eval_data, max_lay_achieved



    def learn(self, num_updates):
        """Update networks for num_updates"""

        learn_history = {}
        learn_history['reward'] = []
        learn_history['actor_loss'] = []
        if self.fw:
            learn_history['fw_bonus'] = []
            learn_history['fw_loss'] = []
        learn_summary = {}

        if self.replay_buffer.size <= 250:
            return learn_summary

        self.critic.train()
        self.actor.train()
        if self.fw:
            self.state_predictor.train()
        for _ in range(num_updates):
            old_states, actions, rewards, new_states, goals, done = self.replay_buffer.get_batch()

            # use forward model to update reward with curiosity
            if self.fw:
                bonus = self.state_predictor.pred_bonus(actions, old_states, new_states).unsqueeze(1)
                eta = self.state_predictor.eta
                rewards = rewards * eta + (1-eta) * bonus
                learn_history['fw_bonus'].append(bonus.mean().item())

            learn_history['reward'] += rewards.cpu().numpy().tolist()

            q_update = self.critic.update(old_states, actions, rewards, new_states, goals, self.actor(new_states, goals).detach(), done)
            actor_loss = -self.critic(old_states, goals, self.actor(old_states, goals)).mean()
            self.actor.update(actor_loss)
            learn_history['actor_loss'] += [actor_loss.detach().item()]

            for k,v in q_update.items():
                if k not in learn_history.keys(): learn_history[k] = []
                learn_history[k].append(v)

            if self.fw:
                learn_history['fw_loss'].append(self.state_predictor.update(old_states, actions, new_states))

        r_vals = [-0.0, -1.0]

        if self.layer_number != 0:
            r_vals.append(float(-self.time_scale))

        for reward_val in r_vals:
            learn_history["reward_{}_frac".format(reward_val)] = float(np.sum(np.isclose(learn_history['reward'], reward_val))) / len(learn_history['reward'])

        for k,v in learn_history.items():
            learn_summary[k] = v.mean() if isinstance(v, torch.Tensor) else np.mean(v)

        return learn_summary
