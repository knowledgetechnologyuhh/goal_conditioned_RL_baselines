import numpy as np
import gym
import pickle

from baselines import logger
from baselines.mbhac.mbhac_policy import MBHACPolicy
from baselines.mbhac.utils import AntWrapper, BlockWrapper, UR5Wrapper
from gym.envs.registration import registry

DEFAULT_ENV_PARAMS = {
    'AntReacherEnv-v0':{ },
}

DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # mbhac
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden_size': 64,  # number of neurons in each hidden layers
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'scope': 'mbhac',  # can be tweaked for testing
    'reuse': False,
    'use_mpi': False,
    'rollout_batch_size': 1,  # per mpi thread
    'atomic_noise': 0.1,
    'subgoal_noise': 0.1
}

POLICY_ACTION_PARAMS = {}
CACHED_ENVS = {}
ROLLOUT_PARAMS = { 'T': 50, 'policy_action_params': {} }
EVAL_PARAMS = { 'policy_action_params': { } }

OVERRIDE_PARAMS_LIST = list(DEFAULT_PARAMS.keys())

ROLLOUT_PARAMS_LIST = ['T', 'rollout_batch_size', 'env_name']


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    # MBHAC params
    mbhac_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)

    kwargs['make_env'] = make_env
    if env_name[:3] == 'Cop':
        registry.env_specs[env_name]._kwargs['tmp'] = 3
        registry.env_specs[env_name]._kwargs['render'] = kwargs['render']
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    if env_name[:3] == 'Cop':
        registry.env_specs[env_name]._kwargs['tmp'] = 1
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']

    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']

    for name in ['hidden_size', 'layers', 'Q_lr', 'pi_lr', 'max_u', 'scope', 'verbose']:
        mbhac_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]

    kwargs['mbhac_params'] = mbhac_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))

def configure_policy(dims, params):
    # Extract relevant parameters.
    mbhac_params = params['mbhac_params']
    input_dims = dims.copy()

    # MBHAC agent
    wrapper_args = (gym.make(params['env_name']).env, params['n_layers'], params['time_scale'], input_dims)
    print('Wrapper Args', *wrapper_args)
    if 'Ant' in params['env_name']:
        env = AntWrapper(*wrapper_args)
    elif 'UR5' in params['env_name']:
        env = UR5Wrapper(*wrapper_args)
    elif 'Block' in params['env_name']:
        env = BlockWrapper(*wrapper_args)
    elif 'Causal' in params['env_name']:
        env = BlockWrapper(*wrapper_args)
    elif 'Hook' in params['env_name']:
        env = BlockWrapper(*wrapper_args)
    elif 'CopReacher' in params['env_name']:
        env = BlockWrapper(*wrapper_args)

    agent_params = {
            "subgoal_test_perc": params['subgoal_test_perc'],
            "subgoal_penalty": -params['time_scale'],
            "atomic_noise": [params['atomic_noise'] for i in range(input_dims['u'])],
            "subgoal_noise": [params['subgoal_noise'] for i in range(len(env.sub_goal_thresholds))],
            "n_layers": params['n_layers'],
            "batch_size": params['train_batch_size'],
            "buffer_size": params['buffer_size'],
            "time_scale": params['time_scale'],
            "hidden_size": mbhac_params['hidden_size'],
            "Q_lr": mbhac_params['Q_lr'],
            "pi_lr": mbhac_params['pi_lr'],
            # forward model
            "model_based": params['model_based'],
            "mb_params": {
                "hidden_size": params['mb_hidden_size'],
                "lr": params['mb_lr'],
                "eta": params['eta'],
                }
            }

    mbhac_params.update({
        'input_dims': input_dims,  # agent takes an input observations
        'T': params['T'],
        'rollout_batch_size': params['rollout_batch_size'],
        'gamma': params['gamma'],
        'reuse': params['reuse'],
        'use_mpi': params['use_mpi'],
        'batch_size': params['train_batch_size'],
        'buffer_size': params['buffer_size'],
        'n_layers': params['n_layers'],
        'agent_params': agent_params,
        'env': env
        })
    mbhac_params['info'] = {
        'env_name': params['env_name'],
    }

    policy = MBHACPolicy(**mbhac_params)
    env.agent = policy

    if params['env_name'][:3] == 'Cop':
        env.reset()

    return policy

def load_policy(restore_policy_file, params):
    # Load policy.
    with open(restore_policy_file, 'rb') as f:
        policy = pickle.load(f)
    return policy

def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())
    if params['env_name'][:3] == 'Cop':
        env.reset()

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
