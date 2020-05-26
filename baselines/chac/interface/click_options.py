import click
chac_options = [
click.option('--n_test_rollouts', type=int, default=25, help='The number of testing rollouts.'),
click.option('--batch_size', type=int, default=1024, help='The number of state transitions processed during network training.'),
click.option('--n_train_batches', type=int, default=40, help='The number of batches to train the networks.'),
click.option('--buffer_size', type=int, default=500, help='The number of rollouts to store in each levels replay buffer.'),

# HAC
click.option('--q_lr', type=float, default=0.001, help='Critic learning rate'),
click.option('--q_hidden_size', type=int, default=64, help='Hidden size used for the critic network'),
click.option('--mu_lr', type=float, default=0.001, help='Actor learning rate'),
click.option('--mu_hidden_size', type=int, default=64, help='Hidden size used for the actor network'),

click.option('--time_scales', type=str, default='27,27', help='Steps per level from lowest to highest also used to define penalties'),
click.option('--subgoal_test_perc', type=float, default=0.3, help='The percentage of subgoals to test.'),
click.option('--n_levels', type=int, default=2, help='Total number hierarchies'),

click.option('--n_pre_episodes', type=int, default=30, help='Number of finished episodes before training networks'),
click.option('--random_action_perc', type=float, default=0.3, help='Percentage of taking random actions'),
click.option('--atomic_noise', type=float, default=0.2, help='Exploration noise added to atomic actions'),
click.option('--subgoal_noise', type=float, default=0.2, help='Exploration noise added to subgoal actions'),

# Forward model
click.option('--fw', type=int, default=0, help='Enable forward model'),
click.option('--fw_hidden_size', type=str, default='128,128,128', help='Size for each hidden layer of the forward model'),
click.option('--fw_lr', type=float, default=0.001, help='Learning rate to train the forward model'),
click.option('--eta', type=float, default=0.5, help='Reward fraction (r_e * eta + (1-eta) * r_i)'),

click.option('--verbose', type=bool, default=False),
click.option('--num_threads', type=int, default=1, help='Number of threads used for intraop parallelism on CPU')
]

def click_main(func):
    for option in reversed(chac_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs