import click

_global_options = [
click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on'),
click.option('--base_logdir', type=str, default='storage2/data/baselines/logs', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/'),
click.option('--n_epochs', type=int, default=300, help='the max. number of training epochs to run'),
click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)'),
click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code'),
click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
]

def click_main(func):
    for option in reversed(_global_options):
        func = option(func)
    return func