import click
_random_options = [
click.option('--n_test_rollouts', type=int, default=0, help='Number of test rollouts'),
click.option('--model_train_batch_size', type=int, default=40, help='The batch size (parallel episodes) for model training.'),
]

def click_main(func):
    for option in reversed(_random_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs