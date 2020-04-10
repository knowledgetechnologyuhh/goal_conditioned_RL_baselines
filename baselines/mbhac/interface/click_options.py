import click
mbhac_options = [
click.option('--network_class', type=str, default='baselines.mbhac.actor_critic:ActorCritic', help='The Neural network model to use.'),
click.option('--n_test_rollouts', type=int, default=10, help='The number of testing rollouts.'),
click.option('--train_batch_size', type=int, default=1024, help='The number of state transitions processed during network training.'),
click.option('--n_train_batches', type=int, default=40, help='The number of batches for model training.'),
click.option('--time_scale', type=int, default=27, help='The steps per layer.'),
click.option('--buffer_size', type=int, default=250, help='The number of rollouts to store per in each layers buffer.'), # old default 500
click.option('--subgoal_test_perc', type=float, default=0.3, help='The percentage of subgoals to test.'),
click.option('--n_layers', type=int, default=2, help='The number of used layers'),
click.option('--model_based', type=int, default=0, help='Model-based flag'),
click.option('--mb_hidden_size', type=str, default='64,64,64', help='Size for each layer added to the forward model'),
click.option('--mb_lr', type=float, default=0.001, help='Learning rate to train the forward model'),
click.option('--eta', type=float, default=0.1, help='Factor to scale the curiosity'),
click.option('--verbose', type=bool, default=False)
]

def click_main(func):
    for option in reversed(mbhac_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs