"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available in the "options.py" file.
"""

import pickle as cpickle
import agent as Agent
from utils import print_summary
from tqdm import tqdm

NUM_BATCH = 50
TEST_FREQ = 2
# TEST_FREQ = 1


# num_test_episodes = 3

def run_HAC(FLAGS,env,agent):

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
            success, eval_data = agent.train(env, episode, total_train_episodes, eval_data)
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
                success, eval_data = agent.train(env, episode, total_train_episodes, eval_data)

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
