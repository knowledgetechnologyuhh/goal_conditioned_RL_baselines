import click
_no_options = [
click.option('--none', type=int, default=0, help='A sample argument')
]

def click_main(func):
    for option in reversed(_no_options):
        func = option(func)
    return func