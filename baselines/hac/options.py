import argparse

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy',
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
        choices=['ANT_FOUR_ROOMS_2', 'ANT_FOUR_ROOMS_3', 'ANT_REACHER_2','ANT_REACHER_2_SMALL_SUBG',
            'ANT_REACHER_2_VEL_GOAL', 'ANT_REACHER_3', 'PENDULUM_LAY_1', 'PENDULUM_LAY_2', 'PENDULUM_LAY_3',
            'UR5_LAY_1', 'UR5_LAY_2', 'UR5_LAY_3'],
        help='The environment to run. The options are the files in the \'env_designs\' subfolder',
        type=str
    )
    parser.add_argument('--early_stop_threshold', type=float, default=0.95, help='The early stopping threshold.')

    parser.add_argument('--base_logdir', type=str, default='hac_logs', help='The directory to store the performance logs.')

    parser.add_argument(
        '--early_stop_data_column', type=str, default='test/subgoal_1_succ_rate',
                 help='The data column on which early stopping is based.'
    )
    #  parser.add_argument('--n_train_rollouts', type=int, default=10, help='The number of training episodes per epoch.')
    #  parser.add_argument('--n_test_rollouts', type=int, default=3, help='The number of test episodes per epoch.')
    #  parser.add_argument('--n_train_batches', type=int, default=40, help='The number of training batches after each epoch.')
    #  parser.add_argument('--batch_size', type=int, default=1024, help='The batch size for training.')
    #  parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs.')
    parser.add_argument('--steps_per_layer', dest='time_scale', type=int, default=30, help='The steps per layer.')
    parser.add_argument('--buffer_size', type=int, default=500, help='The number of rollouts to store per in each layers buffer.')
    parser.add_argument('--timesteps_per_action', type=int, default=0, help='The number of simulation steps per executed action. Set to 0 to use environment default.')
    parser.add_argument('--test_subgoal_perc', type=float, default=0.3, help='The percentage of subgoals to test.')


    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS
