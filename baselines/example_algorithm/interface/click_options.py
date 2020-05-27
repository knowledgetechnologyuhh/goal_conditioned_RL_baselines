import click
_random_options = [
click.option('--n_train_batches', type=int, default=40, help='The number of batches for model training.'),
click.option('--dummy', type=int, default=1, help='This argument is useless. Just for demostration'),
click.option('--n_test_rollouts', type=int, default=10, help='The number of testing rollouts.'),

]

def click_main(func):
    for option in reversed(_random_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs