import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Retrain boolean ("--retrain")
- If included, actor and critic neural network parameters are reset

2. Testing boolean ("--test")
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks.
- If not included, periods of training are by default interleaved with periods of testing to evaluate progress.

3. Show boolean ("--show")
- If included, training will be visualized

4. Train Only boolean ("--train_only")
- If included, agent will be solely in training mode and will not interleave periods of training and testing

5. Verbosity boolean ("--verbose")
- If included, summary of each transition will be printed

6. All Trans boolean ("--all_trans")
- If included, all transitions including (i) hindsight action, (ii) subgoal penalty, (iii) preliminary HER, and (iv) final HER transitions will be printed.  Use below options to print out specific types of transitions.

7. Hindsight Action trans boolean ("hind_action")
- If included, prints hindsight actions transitions for each level

8. Subgoal Penalty trans ("penalty")
- If included, prints the subgoal penalty transitions

9. Preliminary HER trans ("prelim_HER")
-If included, prints the preliminary HER transitions (i.e., with TBD reward and goal components)

10.  HER trans ("HER")
- If included, prints the final HER transitions for each level

11. Show Q-values ("--Q_values")
- Show Q-values for each action by each level

"""

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--all_trans',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--hind_action',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--penalty',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--prelim_HER',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--HER',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--Q_values',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--env',
        choices=['ANT_FOUR_ROOMS_2', 'ANT_FOUR_ROOMS_3', 'ANT_REACHER_2','ANT_REACHER_2_SMALL_SUBG', 'ANT_REACHER_2_VEL_GOAL', 'ANT_REACHER_3', 'PENDULUM_LAY_1', 'PENDULUM_LAY_2', 'PENDULUM_LAY_3', 'UR5_LAY_1', 'UR5_LAY_2', 'UR5_LAY_3'],
        help='The environment to run. The options are the files in the \'env_designs\' subfolder',
        type=str
    )
    parser.add_argument('--early_stop_threshold', type=float, default=0.95, help='The early stopping threshold.')

    parser.add_argument('--base_logdir', type=str, default='hac_logs', help='The directory to store the performance logs.')

    parser.add_argument(
        '--early_stop_data_column', type=str, default='test/subgoal_1_succ_rate',
                 help='The data column on which early stopping is based.'
    )
    parser.add_argument('--n_train_rollouts', type=int, default=10, help='The number of training episodes per epoch.')
    parser.add_argument('--n_test_rollouts', type=int, default=3, help='The number of test episodes per epoch.')
    parser.add_argument('--n_train_batches', type=int, default=40, help='The number of training batches after each epoch.')
    parser.add_argument('--batch_size', type=int, default=1024, help='The batch size for training.')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--steps_per_layer', dest='time_scale', type=int, default=33, help='The steps per layer.')
    parser.add_argument('--buffer_size', type=int, default=600, help='The number of rollouts to store per in each layers buffer.')
    parser.add_argument('--timesteps_per_action', type=int, default=0,
                        help='The number of simulation steps per executed action. Set to 0 to use environment default.')
    parser.add_argument('--test_subgoal_perc', type=float, default=0.3,
                        help='The percentage of subgoals to test.')


    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS
