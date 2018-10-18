import click
_her_options = [
click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped'),
click.option('--network_class', type=str, default='baselines.her.actor_critic:ActorCritic', help='The Neural network model to use.'),
click.option('--replay_k', type=int, default=4, help='The ratio between HER replays and regular replays.'),
click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future',
             help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.'),
click.option('--curriculum_sampling', type=str, default='stochastic3_0.5_5_50_5', help='The curriculum subgoal sampling success rate. Either '
                                                                 '1) \'none\', i.e. the goal mask always uses the full goal or '
                                                                 '2) \'stochastic3_GR_N_P_H\', i.e., stochastically select those goal masks for which the goal is achieved at. '
                                                                     'the success rate is approximated using conditional independence assumption of subgoals. the '
                                                                        'success rate GR \in [0,1] during training. The probability distribution is computed as abs(R-GR)^N, '
                                                                        'where R is the success rate averaged over H last tries.')

]

def click_main(func):
    for option in reversed(_her_options):
        func = option(func)
    return func

@click.command()
@click_main
def get_click_option(**kwargs):
    return kwargs