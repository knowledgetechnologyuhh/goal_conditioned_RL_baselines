from baselines.template.interface.config import *
from baselines.example_algorithm.random_policy import RandomPolicy


def configure_policy(dims, params):

    rollout_batch_size = params['rollout_batch_size']
    policy_params = params['policy_params']
    input_dims = dims.copy()

    policy_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'rollout_batch_size': rollout_batch_size,
                        })
    policy_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = RandomPolicy(**policy_params)
    return policy
