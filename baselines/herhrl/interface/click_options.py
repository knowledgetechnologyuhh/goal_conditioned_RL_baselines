import click
_her_options = [
click.option('--n_test_rollouts', type=int, default=10, help='The number of testing rollouts.'),
click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped'),
click.option('--network_class', type=str, default='baselines.herhrl.actor_critic:ActorCritic',
             help='The Neural network model to use.'),
click.option('--replay_k', type=int, default=4,
             help='The ratio between HER replays and regular replays. Set to 0 for DDPG only.'),
click.option('--train_batch_size', type=int, default=128,
             help='The number of state transitions processed during network training.'),
click.option('--n_train_batches', type=int, default=15, help='The number of batches for model training.'),
click.option('--n_subgoals_layers', type=str, default='[3]', help='The minimal number of subgoals for all layers except the lowest layers. So if you put two values here, you will have three layers.'),
click.option('--policies_layers', type=str, default='[MIX_PDDL_HRL_POLICY]', help='The policies to use for each layer except for the lowest which must always be DDPG_HER_HRL_POLICY. Possible options include MIX_PDDL_HRL_POLICY, DDPG_HER_HRL_POLICY and PDDL_POLICY.'),
click.option('--penalty_magnitude', type=int, default=-2, help='The magnitude of penalty score when subgoal is missed.'),
click.option('--test_subgoal_perc', type=float, default=1.0, help='Percentage of event to test subgoal. 0 mean no '
                                                                  'testing and hence no penalty.'),
click.option('--mix_p_threshold', type=float, default=0.15, help='Switching (between pddl and ddpg) threshold of mix policy'),
click.option('--mix_p_steepness', type=float, default=4.0, help='Steepness of the sigmoid switching function .'),
click.option('--obs_noise_coeff', type=float, default=0.0, help='Fraction of element-wise range of observation to sample from to generate observation noise.')
]

def click_main(func):
    for option in reversed(_her_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs