import click
_her_options = [
click.option('--n_test_rollouts', type=int, default=10, help='The number of testing rollouts.'),
click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped'),
click.option('--network_class', type=str, default='baselines.mbhac.actor_critic:ActorCritic', help='The Neural network model to use.'),
click.option('--replay_k', type=int, default=4, help='The ratio between HER replays and regular replays.'),
click.option('--train_batch_size', type=int, default=256, help='The number of state transitions processed during network training.'),
click.option('--n_train_batches', type=int, default=40, help='The number of batches for model training.'),
click.option('--replay_strategy', default='future',
             type=click.Choice(['future', 'none', 'future_mask', 'now_mask', 'final_mask', 'full_mask']),
             help='The method for transition sampling in hindsight future replay. Either '
                 '0) \'none\': no HER sampling, just use the standard DDPG algorithm.'
                 '1) \'future\': as in the normal future implementation of HER.'
                 '2) \'future_mask\': the original goals are masked with observations from somewhere in the future. I think this is what we used for CGM, but not 100% sure anymore'
                 '3) \'now_mask\': the original goals are masked with observations from the current timestep.'
                 '4) \'final_mask\' : the original goals are masked with observations from the final timestep.'
                 '5) \'full_mask\' : replay transition sampling is completely masked. TODO: Not yet Implemented'),
click.option('--levy_env', is_flag=True, default=False)
]

def click_main(func):
    for option in reversed(_her_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs