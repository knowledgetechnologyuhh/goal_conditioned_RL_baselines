import click
_random_options = [
click.option('--dummy', type=int, default=1, help='This argument is useless. Just for demostration'),

]

def click_main(func):
    for option in reversed(_random_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs