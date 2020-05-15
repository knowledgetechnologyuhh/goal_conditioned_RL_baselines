from baselines.template.interface.config import *

from baselines.herhrl.ddpg_her_hrl_policy import DDPG_HER_HRL_POLICY
from baselines.herhrl.mix_pddl_hrl_policy import MIX_PDDL_HRL_POLICY
from baselines.herhrl.pddl_policy import PDDL_POLICY
from baselines.herhrl.her import make_sample_her_transitions as make_sample_her_transitions_hrl
# from baselines.her.her import make_sample_her_transitions
# from baselines.her_pddl.pddl.pddl_util import obs_to_preds_single
import importlib

DEFAULT_PARAMS['network_class'] = 'baselines.herhrl.actor_critic:ActorCritic'
DEFAULT_PARAMS['buffer_size'] = int(5E3)  # for experience replay
DEFAULT_PARAMS['scope'] = 'ddpg_hrl'
DEFAULT_PARAMS['has_child'] = False

EVAL_PARAMS['noise_eps'] = 0.0
EVAL_PARAMS['random_eps'] = 0.0

OVERRIDE_PARAMS_LIST = ['action_steps', 'policies_layers', 'shared_pi_err_coeff', 'action_l2', 'network_classes']

ROLLOUT_PARAMS_LIST = ['noise_eps', 'random_eps', 'replay_strategy', 'env_name']


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()
    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    ROLLOUT_PARAMS['cached_make_env'] = cached_make_env
    EVAL_PARAMS['cached_make_env'] = cached_make_env
    if env_name[:3] == 'Cop':
        registry.env_specs[env_name]._kwargs['render'] = kwargs['render']

    tmp_env = cached_make_env(kwargs['make_env'])
    action_steps = [int(n_s) for n_s in kwargs['action_steps'][1:-1].split(",") if n_s != '']
    kwargs['action_steps'] = action_steps
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals',
                 'shared_pi_err_coeff']:
        if name in kwargs.keys():
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
    for name in ['replay_strategy', 'replay_k', 'penalty_magnitude', 'has_child']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    sample_her_transitions = make_sample_her_transitions_hrl(**her_params)
    return sample_her_transitions


def configure_policy(dims, params):
    # Extract relevant parameters.
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']
    reuse = params['reuse']
    use_mpi = params['use_mpi']
    p_steepness = params['mix_p_steepness']
    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    subgoal_scale, subgoal_offset = env.env.get_scale_and_offset_for_normalized_subgoal()
    units_per_obs_len = 12
    n_obs = len(env.env._get_obs()['observation'])
    ddpg_params.update({
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'reuse': reuse,
                        'use_mpi': use_mpi,
                        'clip_pos_returns': True,  # clip positive returns for Q-values
                        'h_level': 0,
                        'p_steepness': p_steepness,
                        'hidden': units_per_obs_len * n_obs
    })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }

    n_subgoals = params['action_steps']
    policy_types = [getattr(importlib.import_module('baselines.herhrl.' + (policy_str.lower())), policy_str) for
                    policy_str in params['policies_layers'][1:-1].split(",") if policy_str != '']
    net_classes = [net_class for net_class in params['network_classes'][1:-1].split(",") if net_class != '']
    policies = []
    for l, (n_s, ThisPolicy, net_class) in enumerate(zip(n_subgoals, policy_types, net_classes)):

        if l == (len(n_subgoals) - 1): # If this is the final lowest layer
            input_dims = dims.copy()
            subgoal_scale = np.ones(input_dims['u'])
            subgoal_offset = np.zeros(input_dims['u'])
            has_child = False
        else:
            input_dims = dims.copy()
            input_dims['u'] = input_dims['g']
            has_child = True # penalty only apply for the non-leaf hierarchical layers
        _params = params.copy()
        _params['has_child'] = has_child
        sample_her_transitions = configure_her(_params)
        ddpg_params['sample_transitions'] = sample_her_transitions
        ddpg_params['network_class'] = "baselines.herhrl." + net_class
        this_params = ddpg_params.copy()
        gamma = 1. - 1. / n_s
        this_params.update({'input_dims': input_dims,  # agent takes an input observations
                            'T': n_s,
                            'subgoal_scale': subgoal_scale,
                            'subgoal_offset': subgoal_offset,
                            'h_level': l,
                            'gamma': gamma,
                            'buffer_size': ddpg_params['buffer_size'] * n_s,
                            'clip_return': (1. / (1. - gamma)) if params['clip_return'] else np.inf,
                            })
        this_params['scope'] += '_l_{}'.format(l)
        policy = ThisPolicy(**this_params)
        policies.append(policy)
    if len(policies) > 0:
        h_level_ctr = 1
        for p, p_child in zip(policies[:-1], policies[1:]):
            p.child_policy = p_child
            p.child_policy.h_level = h_level_ctr
            p.child_policy.sess = p.sess
            h_level_ctr += 1

    return policies[0]


def load_policy(restore_policy_file, params):
    # Load policy.
    with open(restore_policy_file, 'rb') as f:
        policy = pickle.load(f)
    # Set sample transitions (required for loading a policy only).
    _params = params.copy()
    policy = set_policy_params(policy, _params)
    return policy


def set_policy_params(policy, params):
    child_params = params.copy()
    if policy.child_policy is None: # Don't use a penalty for the leaf policy
        params['has_child'] = False
    else:
        params['has_child'] = True
    policy.sample_transitions = configure_her(params)
    policy.rollout_batch_size = params['rollout_batch_size']
    if policy.buffer is not None:
        policy.buffer.sample_transitions = policy.sample_transitions
    if policy.child_policy is not None:
        set_policy_params(policy.child_policy, child_params)
    return policy
