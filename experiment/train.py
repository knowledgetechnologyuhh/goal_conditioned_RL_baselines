import wtm_envs.register_envs

import os
import sys
import pickle
import click
import numpy as np
import json
from mpi4py import MPI
import time

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.her.experiment.config import configure_her
from baselines.her.experiment.plot import load_results

from subprocess import CalledProcessError
import subprocess


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    best_success_rate = -1
    success_rates = []
    # if the std dev of the success rate of the last epochs is larger than X do early stopping.
    n_epochs_avg_for_early_stop = 4
    # avg_success_for_early_stop = 0.95
    for epoch in range(n_epochs):
        # train
        logger.info("Training epoch {}".format(epoch))
        rollout_worker.clear_history()
        dur_ro = 0
        dur_train = 0
        dur_start = time.time()
        for cyc in range(n_cycles):
            # logger.info("Performing ")
            ro_start = time.time()
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            dur_ro += time.time() - ro_start
            train_start = time.time()
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()
            dur_train += time.time() - train_start
        dur_total = time.time() - dur_start
        logger.info('Time for epoch {}: {:.2f}. Rollout time: {:.2f}, Training time: {:.2f}'.format(epoch, dur_total, dur_ro, dur_train))

        # eval
        logger.info("Evaluating epoch {}".format(epoch))
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        success_rates.append(success_rate)
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)
        if len(success_rates) > n_epochs_avg_for_early_stop:
            # stddev = np.std(success_rates[-n_epochs_avg_for_early_stop:])
            avg = np.mean(success_rates[-n_epochs_avg_for_early_stop:])
            logger.info('Mean of success rate of last {} epochs: {}'.format(n_epochs_avg_for_early_stop, avg))
            if avg >= kwargs['early_stop_success_rate'] and kwargs['early_stop_success_rate'] != 0:
                logger.info('Policy is good enough now, early stopping')
                break
            # if stddev < n_epochs_avg_for_early_stop:
            #     logger.info(
            #         'Not getting any better, early stopping')
            #     break

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, restore_policy,
    override_params={}, save_policies=True, **kwargs
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['replay_strategy'] = replay_strategy
    params['n_cycles'] = kwargs['n_train_rollout_cycles']
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    if env.find("HLMG") == 0:
        dims = config.configure_masked_dims(params)
    else:
        dims = config.configure_dims(params)

    if restore_policy is None:
        policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    else:
        # Load policy.
        with open(restore_policy, 'rb') as f:
            policy = pickle.load(f)
        # Set sample transitions (required for loading a policy only).
        policy.sample_transitions = configure_her(params)
        policy.buffer.sample_transitions = policy.sample_transitions
        loaded_env_name = policy.info['env_name']
        assert loaded_env_name == env

    # params['n_cycles'] = 5
    rollout_params = {
        'exploit': False,
        # 'use_target_net': False,
        # 'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'render': bool(kwargs['render']),
    }

    eval_params = {
        'exploit': True,
        # 'use_target_net': params['test_with_polyak'],
        # 'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'render': bool(kwargs['render']),
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps', 'goldilocks_sampling', 'mask_at_observation', '_replay_strategy', 'env_name']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    eval_params['training_rollout_worker'] = rollout_worker

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    early_stop_success_rate = kwargs['early_stop_success_rate'] / 100

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, early_stop_success_rate=early_stop_success_rate)
    print("Done training")


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--base_logdir', type=str, default='/storage2/data/baselines/logs', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=300, help='the max. number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
# @click.option('--replay_strategy', type=click.Choice(['future', 'none', 'masked']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')

@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--restore_policy', type=str, default=None, help='The pretrained policy file to start with to avoid learning from scratch again. Useful for interrupting and restoring training sessions.')
@click.option('--network_class', type=str, default='custom_models.actor_critic:ActorCritic', help='The Neural network model to use.')
@click.option('--rollout_batch_size', type=int, default=1, help='The number of simultaneous rollouts.')
@click.option('--n_train_rollout_cycles', type=int, default=50, help='The number of rollout cycles for training.')
@click.option('--n_batches', type=int, default=40, help='The number of batches for training. For each rollout batch, perform gradient descent n_batches times.')
@click.option('--train_batch_size', type=int, default=256, help='The number of state transitions processed during network training.')
@click.option('--render', type=int, default=1, help='Whether or not to render the rollouts.')
@click.option('--replay_k', type=int, default=4, help='The ratio between HER replays and regular replays.')
@click.option('--mask_at_observation', type=int, default=1, help='Whether or not to mask the goal at observation already (0,1)')
@click.option('--early_stop_success_rate', type=int, default=95, help='The required mean success rate  over the last 4 epochs in % to trigger early stopping. 0 for no early stopping')
@click.option('--goldilocks_sampling', type=str, default='none', help='The goldilocks subgoal sampling success rate. Either '
                                                                 '1) \'none\', i.e. the goal mask always uses the full goal or '
                                                                 '2) \'z1toN\', the goals z-positions are masked in 1 out of 5 samples, or'  
                                                                 '3) \'stochastic_R_C\', i.e., stochastically select those goal masks for which the goal is achieved at the '
                                                                        'success rate R \in [0,1] during training'
                                                                        'and emphasize also those masks that require mor goal slots to be achieved by a coefficient C \in [0,1]. '
                                                                        'C=0 means to ignore this emphasizis and C=1 means to ignore the success rates.'
                                                                 '4) \'stochastic2_GR_N_P_H\', i.e., stochastically select those goal masks for which the goal is achieved at the '
                                                                        'success rate GR \in [0,1] during training. The probability distribution is computed as abs(R-GR)^N, '
                                                                        'where R is the success rate averaged over H last tries.'
                                                                        'sample the full goal with probability P \in [0,100].'
                                                                 '5) \'stochastic3_GR_N_P_H\', same as stochasitc2, but here the success rate is approximated using conditional independence assumption of subgoals. '
                                                                      'Useful for goals vectors larger than 6.')

@click.option('--replay_strategy', type=str, default='future', help='The method for transition sampling in hindsight future replay. Either '
                                                                 '0) \'none\': no HER sampling, just use the standard DDPG algorithm.'
                                                                 '1) \'future\': as in the normal future implementation of HER.'
                                                                 '2) \'future_mask\': the original goals are masked with observations from somewhere inthe future.'  
                                                                 '3) \'now_mask\': the original goals are masked with observations from the current timestep.'
                                                                 '4) \'final_mask\' : the original goals are masked with observations from the final timestep.'
                                                                 '5) \'full_mask\' : replay transition sampling is completely masked. TODO: Not yet Implemented')
@click.option('--max_try_idx', type=int, default=199, help='Max. number of tries for this training config.')
@click.option('--try_start_idx', type=int, default=100, help='Index for first try.')

def main(**kwargs):
    kwargs['batch_size'] = kwargs['train_batch_size']
    override_params = ['network_class', 'rollout_batch_size', 'n_batches', 'batch_size', 'goldilocks_sampling', 'replay_k', 'mask_at_observation', 'replay_strategy']
    kwargs['override_params'] = {}
    for op in override_params:
        kwargs['override_params'][op] = kwargs[op]
    subdir_exists = True
    try:
        git_label = str(subprocess.check_output(["git", 'describe', '--always'])).strip()[2:-3]
    except:
        git_label = ''
        print("Could not get git label")
    print("Running training for {}".format(kwargs))
    ctr = kwargs['try_start_idx']
    max_ctr = kwargs['max_try_idx']
    while subdir_exists:
        param_subdir = "_".join(
            ['{}:{}'.format("".join([s[:2] for s in p.split("_")]), str(v).split(":")[-1]) for p, v in
             sorted(kwargs['override_params'].items())]) + "_" + str(ctr)
        if git_label != '':
            logdir = os.path.join(kwargs['base_logdir'], git_label, kwargs['env'], param_subdir)
        else:
            logdir = os.path.join(kwargs['base_logdir'], kwargs['env'], param_subdir)
        subdir_exists = os.path.exists(logdir)
        ctr += 1
    print("Data dir: {}".format(logdir))
    os.makedirs(logdir, exist_ok=False)

    time.sleep(10)

    kwargs['logdir'] = logdir
    kwargs['seed'] = int(time.time())

    # Check if training is necessary. It is not if the last run for this configuration did not achieve at least 40% success rate.
    min_succ_rate = 0.08
    do_train = True
    trial_no = ctr - 1
    print("Trying this config for {}th time. ".format(trial_no))
    last_logdir = "_".join(logdir.split("_")[:-1])+"_{}".format(trial_no - 1)
    last_res_file = last_logdir+"/progress.csv"
    if not os.path.isfile(last_res_file):
        do_train = True
    elif 'n_epochs' not in kwargs.keys():
        do_train = True
    else:
        try:
            last_res = load_results(last_res_file)
            if len(last_res['test/success_rate']) == kwargs['n_epochs']:
                last_succ_rates = np.mean(last_res['test/success_rate'][-4:])
                if last_succ_rates < min_succ_rate:
                    do_train = False
                    print("Last time this did only achieve {} success rate avg over last 4 epochs... skipping a new test!".format(
                        last_succ_rates))
        except:
            print("Could not load progress data {}".format(last_res_file))
    if trial_no > max_ctr:
        print("Already collected enough data for this parameterization")
        do_train = False
    # if subdir_exists:
    #     do_train = False
    if do_train:
        print("Launching training")
        launch(**kwargs)

if __name__ == '__main__':
    main()