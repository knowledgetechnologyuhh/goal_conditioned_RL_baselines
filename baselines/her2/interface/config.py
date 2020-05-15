from baselines.template.interface.config import *

from baselines.her.ddpg import DDPG
from baselines.her.her import make_sample_her_transitions

DEFAULT_PARAMS['network_class'] = 'baselines.her.actor_critic:ActorCritic'

ROLLOUT_PARAMS['T'] = 50
EVAL_PARAMS['T'] = 50

OVERRIDE_PARAMS_LIST = ['network_class', 'rollout_batch_size', 'n_batches', 'batch_size', 'replay_k','replay_strategy']

ROLLOUT_PARAMS_LIST = ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps', '_replay_strategy', 'env_name']


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    # create one environment and make it available for later use
    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    ROLLOUT_PARAMS['cached_make_env'] = cached_make_env
    EVAL_PARAMS['cached_make_env'] = cached_make_env
    if env_name[:3] == 'Cop':
        registry.env_specs[env_name]._kwargs['render'] = kwargs['render']

    env = cached_make_env(kwargs['make_env'])
    assert hasattr(env, '_max_episode_steps')
    kwargs['T'] = env._max_episode_steps
    env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def configure_policy(dims, params):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']
    reuse = params['reuse']
    use_mpi = params['use_mpi']
    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    if params['env_name'][:3] != 'Cop':
        env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if params['clip_return'] else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'reuse': reuse,
                        'use_mpi': use_mpi,
                        # 'n_preds' : 0,
                        # 'h_level' : 0,
                        # 'subgoal_scale': [1,1,1,1],
                        # 'subgoal_offset': [0, 0, 0, 0],
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(**ddpg_params)

    return policy


def load_policy(restore_policy_file, params):
    # Load policy.
    with open(restore_policy_file, 'rb') as f:
        policy = pickle.load(f)
    # Set sample transitions (required for loading a policy only).
    policy.sample_transitions = configure_her(params)
    policy.buffer.sample_transitions = policy.sample_transitions
    return policy
