import numpy as np
import gym
import pickle

from baselines import logger
from baselines.herhrl.ddpg import DDPG_HRL as DDPG_HRL
from baselines.her.ddpg import DDPG as DDPG
from baselines.herhrl.her import make_sample_her_transitions as make_sample_her_transitions_hrl
from baselines.her.her import make_sample_her_transitions
# from baselines.her_pddl.pddl.pddl_util import obs_to_preds_single

DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 20
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.herhrl.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg_hrl',  # can be tweaked for testing
    'relative_goals': False,
    # ddpg get actions
    'reuse': False,
    'use_mpi': True,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # 'random_eps': 0.05,  # percentage of time a random action is taken
    # 'noise_eps': 0.05,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # hrl
    'n_subgoals': [2]    # added for Hierarchical RL. Subgoals per layer, except for the lowest layer.
}

POLICY_ACTION_PARAMS = {

    }

CACHED_ENVS = {}

ROLLOUT_PARAMS = {
        'use_demo_states': True,
        'T': 50,
        'policy_action_params': {'exploit': False,
                                 'compute_Q': False,
                                 'noise_eps': 0.2,
                                 'random_eps': 0.3,
                                 'use_target_net': False}
    }

EVAL_PARAMS = {
        'use_demo_states': False,
        'T': 50,
        'policy_action_params': {'exploit': True,
                                 'compute_Q': True,
                                 'noise_eps': 0.2,
                                 'random_eps': 0.3,
                                 'use_target_net': False
                                 # 'use_target_net': params['test_with_polyak'],
                                 }
    }
"""
compute_Q=self.compute_Q,
noise_eps=self.noise_eps if not self.exploit else 0.,
random_eps=self.random_eps if not self.exploit else 0.,
use_target_net=self.use_target_net)
"""

OVERRIDE_PARAMS_LIST = ['network_class', 'rollout_batch_size', 'n_batches', 'batch_size', 'replay_k','replay_strategy']

ROLLOUT_PARAMS_LIST = ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps', '_replay_strategy', 'env_name',
                       'n_subgoals']


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
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
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


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params, hrl=True):
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
    if hrl:
        sample_her_transitions = make_sample_her_transitions_hrl(**her_params)
    else:
        sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


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
    env.reset()
    preds = env.env.get_preds()
    n_preds = len(preds[0])
    ddpg_params.update({
                        'T': params['T'],
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'gamma': gamma,
                        'reuse': reuse,
                        'use_mpi': use_mpi,
                        'n_preds': n_preds,
                        'sample_transitions': sample_her_transitions,
                        'clip_pos_returns': True,  # clip positive returns for Q-values
                        'clip_return': (1. / (1. - gamma)) if params['clip_return'] else np.inf,  # max abs of return
    })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }

    t_remaining = params['T']
    policies = []
    for l, n_s in enumerate(params['n_subgoals'] + [None]):
        if n_s is None: # If this is the final lowest layer
            input_dims = dims.copy()
            n_s = t_remaining
            # max_u = 1.0
            # u_offset = np.array([0, 0, 0, 0])
        else:
            input_dims = dims.copy()
            input_dims['u'] = input_dims['g']
            # max_u = 0.15 # TODO: This pertains to the tower building environment. Should be set somewhere else in the environment
            #             # u_offset = np.array([1.0, 0.27, 0.515, 1.0, 0.27, 0.515]) definition...
        this_params = ddpg_params.copy()
        this_params.update({'input_dims': input_dims,  # agent takes an input observations
                            'T': n_s
                            })
        t_remaining = int(t_remaining / n_s)
        this_params['scope'] += '_l_{}'.format(l)
        policy = DDPG_HRL(**this_params)
        policies.append(policy)
    return policies

def load_policy(restore_policy_file, params):
    # Load policy.
    with open(restore_policy_file, 'rb') as f:
        policy = pickle.load(f)
    # Set sample transitions (required for loading a policy only).
    policy.sample_transitions = configure_her(params)
    policy.buffer.sample_transitions = policy.sample_transitions
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
