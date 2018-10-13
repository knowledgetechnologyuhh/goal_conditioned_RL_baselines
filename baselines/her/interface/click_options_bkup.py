import click
_her_options = [
click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped'),
click.option('--restore_policy', type=str, default=None, help='The pretrained policy file to start with to avoid learning from scratch again. Useful for interrupting and restoring training sessions.'),
click.option('--network_class', type=str, default='baselines.her.actor_critic:ActorCritic', help='The Neural network model to use.'),
click.option('--rollout_batch_size', type=int, default=1, help='The number of simultaneous rollouts.'),
click.option('--n_train_rollout_cycles', type=int, default=50, help='The number of rollout cycles for training.'),
click.option('--n_batches', type=int, default=40, help='The number of batches for training. For each rollout batch, perform gradient descent n_batches times.'),
click.option('--train_batch_size', type=int, default=256, help='The number of state transitions processed during network training.'),
click.option('--render', type=int, default=1, help='Whether or not to render the rollouts.'),
click.option('--replay_k', type=int, default=4, help='The ratio between HER replays and regular replays.'),
click.option('--mask_at_observation', type=int, default=1, help='Whether or not to mask the goal at observation already (0,1)'),
click.option('--early_stop_success_rate', type=int, default=95, help='The required mean success rate  over the last 4 epochs in % to trigger early stopping. 0 for no early stopping'),
click.option('--goldilocks_sampling', type=str, default='none', help='The goldilocks subgoal sampling success rate. Either '
                                                                 '1) \'none\', i.e. the goal mask always uses the full goal or '
                                                                 '2) \'z1toN\', the goals z-positions are masked in 1 out of 5 samples, or'  
                                                                 '3) \'stochastic_R_C\', i.e., stochastically select those goal masks for which the goal is achieved at the '
                                                                        'success rate R \in [0,1] during training'
                                                                        'and emphasize also those masks that require mor goal slots to be achieved by a coefficient C \in [0,1]. '
                                                                        'C=0 means to ignore this emphasizis and C=1 means to ignore the success rates.'
                                                                 '4) \'stochastic2_GR_N_P_H\', i.e., stochastically select those goal masks for which the goal is achieved at the '
                                                                        'success rate GR \in [0,1] during training. The probability distribution is computed as abs(R-GR)^N, '
                                                                        'where R is the success rate averaged over H last tries.'
                                                                        'sample the full goal with probability P \in [0,100].'
                                                                 '5) \'stochastic3_GR_N_P_H\', same as stochasitc2, but here the success rate is approximated using conditional independence assumption of subgoals. '
                                                                      'Useful for goals vectors larger than 6.'),

click.option('--replay_strategy', type=str, default='future', help='The method for transition sampling in hindsight future replay. Either '
                                                                 '0) \'none\': no HER sampling, just use the standard DDPG algorithm.'
                                                                 '1) \'future\': as in the normal future implementation of HER.'
                                                                 '2) \'future_mask\': the original goals are masked with observations from somewhere inthe future.'  
                                                                 '3) \'now_mask\': the original goals are masked with observations from the current timestep.'
                                                                 '4) \'final_mask\' : the original goals are masked with observations from the final timestep.'
                                                                 '5) \'full_mask\' : replay transition sampling is completely masked. TODO: Not yet Implemented'),
click.option('--max_try_idx', type=int, default=199, help='Max. number of tries for this training config.'),
click.option('--try_start_idx', type=int, default=100, help='Index for first try.')
]

def click_main(func):
    for option in reversed(_her_options):
        func = option(func)
    return func