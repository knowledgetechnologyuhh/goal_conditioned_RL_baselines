import numpy as np
import gym
import torch
import pickle

from baselines import logger
from baselines.chac.chac_policy import CHACPolicy
from baselines.chac.utils import prepare_env
from gym.envs.registration import registry

DEFAULT_ENV_PARAMS = {
    'AntReacherEnv-v0': {},
}

DEFAULT_PARAMS = {
    # chac
    'fw': 1,
    'fw_hidden_size': '64,64,64',
    'eta': 0.5,
    'n_levels': 2,
    'time_scales': '27,27',
    'use_mpi': False,
    'rollout_batch_size': 1,  # per mpi thread
    'atomic_noise': 0.1,
    'subgoal_noise': 0.1
}

POLICY_ACTION_PARAMS = {}
CACHED_ENVS = {}
ROLLOUT_PARAMS = {'T': 0, 'policy_action_params': {}}
EVAL_PARAMS = {'policy_action_params': {}}

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
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    kwargs['chac_params'] = chac_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_policy(dims, params):
    # Extract relevant parameters.
    chac_params = params['chac_params']
    input_dims = dims.copy()

    torch.set_num_threads(params['num_threads'])
    time_scales = np.array([int(t) for t in params['time_scales'].split(',')])
    assert len(time_scales) == params['n_levels']

    # CHAC agent
    env = prepare_env(params['env_name'], time_scales, input_dims)

    agent_params = {
        "subgoal_test_perc": params['subgoal_test_perc'],
        "subgoal_penalties": -1. * time_scales,
        "atomic_noise": [params['atomic_noise'] for i in range(input_dims['u'])],
        "subgoal_noise": [params['subgoal_noise'] for i in range(len(env.sub_goal_thresholds))],
        "n_levels": params['n_levels'],
        "batch_size": params['batch_size'],
        "buffer_size": params['buffer_size'],
        "time_scales": time_scales,
        "q_lr": params['q_lr'],
        "q_hidden_size": params['q_hidden_size'],
        "mu_lr": params['mu_lr'],
        "mu_hidden_size": params['mu_hidden_size'],
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
        'agent_params': agent_params,
        'env': env,
        'verbose': params['verbose'],
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
    # Load policy
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
