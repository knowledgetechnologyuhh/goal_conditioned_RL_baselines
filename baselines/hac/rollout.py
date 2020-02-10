import numpy as np
import time

from baselines.template.util import store_args, logger
import pickle
from mujoco_py import MujocoException
from baselines.template.rollout import Rollout
from tqdm import tqdm
from baselines.hac.utils import print_summary
import sys
import baselines.hac.env_designs
from baselines.hac.options import parse_options
from baselines.hac.utils import EnvWrapper, check_envs, check_validity
import os,sys,inspect
import importlib

class RolloutWorker(Rollout):


    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, history_len=100, render=False, **kwargs):
        Rollout.__init__(self, make_env, policy, dims, logger, T, rollout_batch_size=rollout_batch_size, history_len=history_len, render=render, **kwargs)
        self.graph = kwargs['graph']


    def generate_rollouts_update(self, n_episodes, n_train_batches):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir)
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        FLAGS = parse_options()
        FLAGS.mix_train_test = True
        FLAGS.retrain = True
        FLAGS.Q_values = True
        FLAGS.show = True
        env_import_name = "baselines.hac.env_designs.ANT_FOUR_ROOMS_2_design_agent_and_env"
        design_agent_and_env_module = importlib.import_module(env_import_name)
        # simple tag for agent's tf scope
        FLAGS.id = 0
        # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
        agent, env = design_agent_and_env_module.design_agent_and_env(FLAGS)

        # Print task summary
        print_summary(FLAGS,env)

        total_train_episodes = 0
        total_train_steps = 0
        total_test_episodes = 0
        total_test_steps = 0

        # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
        mix_train_test = False
        if not FLAGS.test and not FLAGS.train_only:
            mix_train_test = True
        # agent.other_params["num_exploration_episodes"] = 3
        num_train_episodes = agent.FLAGS.n_train_rollouts
        num_test_episodes = agent.FLAGS.n_test_rollouts

        for batch in range(agent.FLAGS.n_epochs):

            successful_train_episodes = 0
            successful_test_episodes = 0
            print("\n--- TRAINING epoch {}---".format(batch))
            agent.FLAGS.test = False
            # Evaluate policy every TEST_FREQ batches if interleaving training and testing
            eval_data = {}

            for episode in tqdm(range(num_train_episodes)):

                if agent.FLAGS.verbose:
                    print("\nBatch %d, Episode %d" % (batch, episode))

                # Train for an episode
                success, eval_data, _ = agent.train(env, episode, total_train_episodes, eval_data)
                if success:
                    if agent.FLAGS.verbose:
                        print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                    # Increment successful episode counter if applicable
                    successful_train_episodes += 1

                total_train_episodes += 1
                total_train_steps += agent.steps_taken

            # Save agent
            agent.save_model(batch)
            eval_data['train/total_episodes'] = total_train_episodes
            eval_data['train/epoch_episodes'] = num_train_episodes
            # Finish evaluating policy if tested prior batch
            if mix_train_test:
                print("\n--- TESTING epoch {}---".format(batch))
                agent.FLAGS.test = True
                for episode in tqdm(range(num_test_episodes)):
                    # Train for an episode
                    success, eval_data, _ = agent.train(env, episode, total_train_episodes, eval_data)

                    if success:
                        if agent.FLAGS.verbose:
                            print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                        # Increment successful episode counter if applicable
                        successful_test_episodes += 1

                    # if FLAGS.train_only or (mix_train_test and batch % TEST_FREQ != 0):
                    total_test_episodes += 1
                    total_test_steps += agent.steps_taken
                # Log performance
                success_rate = 0
                if num_test_episodes > 0:
                    success_rate = successful_test_episodes / num_test_episodes
                if agent.FLAGS.verbose:
                    print("\nTesting Success Rate %.2f%%" % success_rate)
                eval_data['test/total_episodes'] = total_test_episodes
                eval_data['test/epoch_episodes'] = num_test_episodes
                eval_data = agent.prepare_eval_data_for_log(eval_data)
                agent.log_performance(success_rate, eval_data, steps=total_train_steps, episode=total_train_episodes, batch=batch)

                print("\n--- END TESTING ---\n")
                early_stop_col = FLAGS.early_stop_data_column
                if early_stop_col in eval_data.keys():
                    early_stop_val = eval_data[early_stop_col]
                    if FLAGS.early_stop_threshold <= early_stop_val:
                        break
                else:
                    print("Warning, early stop column not in keys")

                for k,v in eval_data.items():
                    gap = max(1, 30 - len(k))
                    gap_str = " " * gap
                    print("{}: {} {:.2f}".format(k, gap_str, v))



    def generate_rollouts(self, return_states=False):
        ret = None
        return ret

    def current_mean_Q(self):
        return np.mean(self.custom_histories[0])

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]

        return logger(logs, prefix)

    def save_policy(self, path):
        pass
        #  TODO: Transfer Agent to actual polic file #
        #  with open(path, 'wb') as f:
        #      pickle.dump(self.agent, f)


