import click
mbhac_options = [
click.option('--n_test_rollouts', type=int, default=10, help='The number of testing rollouts.'),
click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped'),
click.option('--network_class', type=str, default='baselines.mbhac.actor_critic:ActorCritic', help='The Neural network model to use.'),
click.option('--replay_k', type=int, default=4, help='The ratio between HER replays and regular replays.'),
click.option('--train_batch_size', type=int, default=1024, help='The number of state transitions processed during network training.'),
click.option('--n_train_batches', type=int, default=40, help='The number of batches for model training.'),
click.option('--steps_per_layer', 'time_scale', type=int, default=33, help='The steps per layer.'),
click.option('--buffer_size', type=int, default=250, help='The number of rollouts to store per in each layers buffer.'), # old default 500
click.option('--timesteps_per_action', type=int, default=0, help='The number of simulation steps per executed action. Set to 0 to use environment default.'),
click.option('--subgoal_test_perc', type=float, default=0.3, help='The percentage of subgoals to test.'),
click.option('--n_layers', type=int, default=2, help='The number of used layers'),
click.option('--model_based', type=int, default=0, help='Model-based flag'),
click.option('--mb_hidden_size', type=str, default='64,64,64', help='Size for each layer added to the forward model'),
click.option('--mb_lr', type=float, default=0.001, help='Learning rate to train the forward model'),
click.option('--eta', type=float, default=0.1, help='Factor to scale the curiosity')
]

def click_main(func):
    for option in reversed(mbhac_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs