import click
_her_options = [
click.option('--n_test_rollouts', type=int, default=10, help='The number of testing rollouts.'),
click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped'),
click.option('--network_class', type=str, default='baselines.herhrl.actor_critic:ActorCritic',
             help='The Neural network model to use.'),
click.option('--replay_k', type=int, default=4,
             help='The ratio between HER replays and regular replays. Set to 0 for DDPG only.'),
click.option('--train_batch_size', type=int, default=256,
             help='The number of state transitions processed during network training.'),
click.option('--n_train_batches', type=int, default=40, help='The number of batches for model training.'),
click.option('--n_subgoals_layers', type=str, default='[2]', help='The minimal number of subgoals for all layers except the lowest layers. So if you put two values here, you will have three layers.')
]

def click_main(func):
    for option in reversed(_her_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs