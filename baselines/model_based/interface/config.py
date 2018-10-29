import numpy as np
import gym
import pickle

from baselines import logger
from baselines.model_based.mb_policy import MBPolicy

DEFAULT_ENV_PARAMS = {}

DEFAULT_PARAMS = {
    # training
    'n_train_rollouts': 50,  # training rollouts per epoch per rollout batch
    'rollout_batch_size': 1,  # batches (parallel rollouts) per mpi thread
    # 'n_batches': 40,  # training batches per cycle for neural network training
    # 'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 0,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'model_buffer_size': 1200, # number of rollouts to store for model training
    'max_u': 1.,  # max absolute value of actions on different coordinates
    'model_network_class': 'baselines.model_based.model_rnn:ModelRNN',
    'scope': 'mbpolicy',
    'model_lr': 0.001,
    'model_train_batch_size': 10,
    'adaptive_model_lr': 0,
}

# POLICY_ACTION_PARAMS = {
#
#     }

CACHED_ENVS = {}

ROLLOUT_PARAMS = {
        'policy_action_params': {}
    }
#
EVAL_PARAMS = {
        'policy_action_params': {}
    }

OVERRIDE_PARAMS_LIST = list(DEFAULT_PARAMS.keys())
#
ROLLOUT_PARAMS_LIST = ['T', 'rollout_batch_size', 'env_name']
# ROLLOUT_PARAMS_LIST = []


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
    # policy params
    policy_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    for name in ['model_buffer_size', 'model_network_class', 'max_u', 'scope', 'model_lr', 'model_train_batch_size']:
        policy_params[name] = DEFAULT_PARAMS[name]
    kwargs['policy_params'] = policy_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_policy(dims, params):

    rollout_batch_size = params['rollout_batch_size']
    policy_params = params['policy_params']
    input_dims = dims.copy()

    env = cached_make_env(params['make_env'])
    env.reset()
    policy_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'rollout_batch_size': rollout_batch_size,
                        'env': env.env,
                        })
    policy_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = MBPolicy(**policy_params)
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
