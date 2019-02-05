import click
_random_options = [
click.option('--n_test_rollouts', type=int, default=0, help='Number of test rollouts'),
click.option('--n_train_batches', type=int, default=40, help='The number of batches for model training.'),
click.option('--model_train_batch_size', type=int, default=40, help='The batch size (parallel episodes) for model training.'),
click.option('--model_lr', type=float, default=0.001, help='The initial learning rate.'),
click.option('--adaptive_model_lr', type=int, default=0, help='Whether or not the learning rate is adaptive.'),
click.option('--model_network_class', type=str, default='baselines.model_based.model_rnn:ModelRNN', help='The network model class to use for the forward model.'),
click.option('--buff_sampling', type=str, default='random',
                # choices=['random', 'max_loss_pred_err', 'mean_loss_pred_err', 'max_loss', 'mean_loss']),
                help='The method to sample from the replay buffer.'
             ),
click.option('--memval_method', type=str, default='uniform',
             help='The method to assess the memory value of each rollout. The memory value is important for forgetting '
                  'training rollouts, i.e., those with the lowest memory value have a high probability to be '
                  'forgotten.'
             # type=click.Choice(['uniform', 'mean_obs_loss', 'max_obs_loss'])
            ),
click.option('--action_selection', default='random',
             help='The method to select actions.'
             # type=click.Choice(['random', 'max_pred_surprise'])
            )
]

def click_main(func):
    for option in reversed(_random_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs