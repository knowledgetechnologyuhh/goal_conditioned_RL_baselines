import click
_random_options = [
click.option('--n_test_rollouts', type=int, default=2, help='Number of test rollouts'),
click.option('--n_model_batches', type=int, default=40, help='The number of batches for model training.'),
]

def click_main(func):
    for option in reversed(_random_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs