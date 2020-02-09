import numpy as np
from baselines.hac.layer import Layer
from baselines.hac.environment import Environment
import pickle as cpickle
import tensorflow as tf
import os
from datetime import datetime
import json
import time
from baselines.util import get_git_label


# Below class instantiates an agent
class Agent():
    def __init__(self,FLAGS, env, agent_params):

        self.FLAGS = FLAGS
        timestamp = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
        git_label = get_git_label()
        self.model_dir = '{}/{}/{}/{}'.format(FLAGS.base_logdir,git_label,self.FLAGS.env,timestamp)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.performance_txt_file = self.model_dir + "/progress.csv".format(self.FLAGS.env)
        self.params_json_file = self.model_dir + "/params.json".format(self.FLAGS.env)
        if not os.path.isfile(self.params_json_file):
            with open(self.params_json_file,'w') as json_file:
                params = vars(FLAGS)
                params['env_name'] = params['env']
                json.dump(params, json_file)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create agent with number of levels specified by user
        self.layers = [Layer(i,FLAGS,env,self.sess,agent_params) for i in range(FLAGS.layers)]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_loc = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()

        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = self.FLAGS.n_train_batches

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params


    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self,env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

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
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved


    def initialize_networks(self):

        model_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(model_vars)

        # Set up location for saving models
        self.model_loc = self.model_dir + '/HAC.ckpt'

        # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if self.FLAGS.retrain == False:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))


    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)


    # Update actor and critic networks for each layer
    def learn(self):
        learn_summaries = []

        for i in range(len(self.layers)):
            learn_summay = self.layers[i].learn(self.num_updates)
            learn_summaries.append(learn_summay)

        return learn_summaries


    # Train agent for an episode
    def train(self,env, episode_num, total_episodes, eval_data):
        start_time = time.time()

        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.FLAGS.layers - 1] = env.get_next_goal(self.FLAGS.test)
        env.display_end_goal(self.goal_array[self.FLAGS.layers - 1])

        if self.FLAGS.verbose:
            print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env.reset_sim(self.goal_array[self.FLAGS.layers - 1])

        if isinstance(self.current_state, dict) and 'observation' in self.current_state.keys():
            self.current_state = self.current_state['observation']

        if env.name == "ant_reacher.xml":
            if self.FLAGS.verbose:
                print("Initial Ant Position: ", self.current_state[:3])
        # print("Initial State: ", self.current_state)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, eval_data, max_lay_achieved = self.layers[self.FLAGS.layers-1].\
            train(self, env, episode_num=episode_num, eval_data=eval_data)

        # Update actor/critic networks if not testing
        if not self.FLAGS.test:
        # if not self.FLAGS.test and total_episodes > 30:
            learn_summaries = self.learn()
            for l in range(self.FLAGS.layers):
                learn_summary = learn_summaries[l]
                for k,v in learn_summary.items():
                    eval_data["train_{}/avg_{}".format(l,k)] = v

        duration = time.time() - start_time
        # Return whether end goal was achieved
        return goal_status[self.FLAGS.layers-1], eval_data, duration


    # Save performance evaluations
    def prepare_eval_data_for_log(self, eval_data):
        for i in range(10):
            for prefx in ['train', 'test']:
                layer_prefix = '{}_{}/'.format(prefx, i)
                if "{}subgoal_succ".format(layer_prefix) in eval_data.keys():
                    subg_succ_rate = eval_data["{}subgoal_succ".format(layer_prefix)] / eval_data["{}n_subgoals".format(layer_prefix)]
                    eval_data['{}subgoal_succ_rate'.format(layer_prefix)] = subg_succ_rate
                if "{}Q".format(layer_prefix) in eval_data.keys():
                    if "{}n_subgoals".format(layer_prefix) in eval_data.keys():
                        n_qvals = eval_data[
                            "{}n_subgoals".format(layer_prefix)]
                    else:
                        n_qvals = 1
                    avg_q = eval_data["{}Q".format(layer_prefix)] / n_qvals
                    eval_data["{}avg_Q".format(layer_prefix)] = avg_q

        if not os.path.isfile(self.performance_txt_file):
            with open(self.performance_txt_file, "w") as txt_logfile:
                txt_logfile.write(
                    "{},{},{},{},".format("DateTime", "epoch", "episode", "train/steps"))
                for k,v in sorted(eval_data.items()):
                    txt_logfile.write("{},".format(k))
                txt_logfile.write("{}\n".format("test/success_rate"))
        return eval_data


    def log_performance(self, success_rate, eval_data, steps=None, episode=None, batch=None):

        # Add latest success_rate to list
        self.performance_log.append(success_rate)

        # Save log
        cpickle.dump(self.performance_log,open(self.model_dir + "/performance_log.p","wb"))

        log_str = ""
        if batch is not None:
            log_str += "{},".format(batch)
        if episode is not None:
            log_str += "{},".format(episode)
        if steps is not None:
            log_str += "{},".format(steps)
        if log_str != "":
            for k,v in sorted(eval_data.items()):
                log_str += "{},".format(v)

            log_str += "{}".format(success_rate)
            # datetime object containing current date and time
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            log_str = "{},{}\n".format(dt_string, log_str)
            with open(self.performance_txt_file, "a") as txt_logfile:
                txt_logfile.write(log_str)

    def logs(self, prefix=''):
        eval_data = self.eval_data
        logs = []

        for k,v in sorted(eval_data.items()):
            logs += [(k , v)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
