import numpy as np
import gym
import pickle

from baselines import logger
from baselines.chac.chac_policy import CHACPolicy
from baselines.chac.utils import AntWrapper, BlockWrapper, UR5Wrapper
from gym.envs.registration import registry

DEFAULT_ENV_PARAMS = {
    'AntReacherEnv-v0':{ },
}

DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # chac
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden_size': 64,  # number of neurons in each hidden layers
    'scope': 'chac',  # can be tweaked for testing
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
    # CHAC params
    chac_params = dict()

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

    for name in ['hidden_size', 'layers', 'max_u', 'scope', 'verbose']:
        chac_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]

    kwargs['chac_params'] = chac_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))

def configure_policy(dims, params):
    # Extract relevant parameters.
    chac_params = params['chac_params']
    input_dims = dims.copy()

    # CHAC agent
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
            "hidden_size": chac_params['hidden_size'],
            "q_lr": params['q_lr'],
            "pi_lr": params['pi_lr'],
            # forward model
            "fw": params['fw'],
            "fw_params": {
                "hidden_size": params['fw_hidden_size'],
                "lr": params['fw_lr'],
                "eta": params['eta'],
                }
            }

    chac_params.update({
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
    chac_params['info'] = {
        'env_name': params['env_name'],
    }

    policy = CHACPolicy(**chac_params)
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
