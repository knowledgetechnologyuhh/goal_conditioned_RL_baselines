import click
_policy_options = [
click.option('--none', type=int, default=0, help='A sample argument')
]

def click_main(func):
    for option in reversed(_policy_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs