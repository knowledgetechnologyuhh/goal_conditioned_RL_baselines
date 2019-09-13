import tensorflow as tf
import os
import sys
this_path =os.getcwd()
sys.path.append(this_path)
os.chdir(this_path)
import click
import numpy as np
import json
from mpi4py import MPI
import time
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
from baselines.util import mpi_fork, get_git_label
import experiment.click_options as main_linker
from subprocess import CalledProcessError
import subprocess
from baselines.util import physical_cpu_core_count, get_subdir_by_params
import wtm_envs.register_envs
from queue import deque


num_cpu = 0

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_episodes, n_train_batches, policy_save_interval,
          save_policies, **kwargs):
    global num_cpu
    rank = MPI.COMM_WORLD.Get_rank()
    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    best_success_rate = -np.inf
    best_early_stop_val = -np.inf
    success_rates = []
    # if the std dev of the success rate of the last epochs is larger than X do early stopping.
    n_epochs_avg_for_early_stop = 4
    early_stop_vals = deque(maxlen=n_epochs_avg_for_early_stop)

    done_training = False
    for epoch in range(n_epochs):
        # train
        logger.info("Training epoch {}".format(epoch))
        rollout_worker.clear_history()
        policy, time_durations = rollout_worker.generate_rollouts_update(n_episodes, n_train_batches)
        logger.info('Time for epoch {}: {:.2f}. Rollout time: {:.2f}, Training time: {:.2f}'.format(epoch, time_durations[0], time_durations[1], time_durations[2]))

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
        for key, val in policy.logs('policy'):
            logger.record_tabular(key, mpi_average(val))

        success_rate = mpi_average(evaluator.current_success_rate())
        success_rates.append(success_rate)

        early_stop_current_val = logger.getkvs()[kwargs['early_stop_data_column']]
        # print("Rank {} esv: {}".format(rank, early_stop_current_val))
        early_stop_vals.append(early_stop_current_val)

        if rank == 0:
            try:
                rollout_worker.policy.draw_hists(img_dir=logger.get_dir())
            except Exception as e:
                pass

            logger.info("Data_dir: {}".format(logger.get_dir()))
            logger.dump_tabular()

            # save latest policy
            evaluator.save_policy(latest_policy_path)

            if policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
                policy_path = periodic_policy_path.format(epoch)
                logger.info('Saving periodic policy to {} ...'.format(policy_path))
                evaluator.save_policy(policy_path)

            # save the policy if it's better than the previous ones
            if kwargs['early_stop_data_column'] is None:
                if success_rate >= best_success_rate and save_policies:
                    best_success_rate = success_rate
                    logger.info(
                        'New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
                    evaluator.save_policy(best_policy_path)
            else:
                assert early_stop_current_val is not None,"Early stopping value should not be none."
                if early_stop_current_val >= best_early_stop_val and save_policies:
                    best_early_stop_val = early_stop_current_val
                    logger.info(
                        'New best value for {}: {}. Saving policy to {} ...'.format(kwargs['early_stop_data_column'], early_stop_current_val, best_policy_path))
                    evaluator.save_policy(best_policy_path)

        if len(early_stop_vals) >= n_epochs_avg_for_early_stop:
            avg = np.mean(early_stop_vals)
            logger.info('Mean of {} of last {} epochs: {}'.format(kwargs['early_stop_data_column'],
                                                                  n_epochs_avg_for_early_stop, avg))

            if avg >= kwargs['early_stop_threshold'] and avg >= kwargs['early_stop_threshold'] != 0:
                logger.info('Policy is good enough now, early stopping')
                done_training = True
                # break

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]
        if (epoch + 1) == n_epochs:
            logger.info('All epochs are finished. Stopping the training now.')
            done_training = True
        if done_training:
            break

def launch(
    env, logdir, n_epochs, num_cpu, seed, policy_save_interval, restore_policy, override_params={}, save_policies=True, **kwargs):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        # whoami = mpi_fork(num_cpu)
        n_cpus_available = physical_cpu_core_count()
        if n_cpus_available < num_cpu:
            whoami = mpi_fork(num_cpu) # This significantly reduces performance!
            assert kwargs['bind_core'] == 0, "Too high CPU count when trying to bind MPI workers to core. You require {} CPUs but have only {}".format(num_cpu, n_cpus_available)
        else:
            if kwargs['bind_core']:
                whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
            else:
                whoami = mpi_fork(num_cpu)  # This significantly reduces performance!
            # try:
            #
            # except CalledProcessError:
            #     # fancy version of mpi call failed, try simple version
            #     whoami = mpi_fork(num_cpu) # This significantly reduces performance!
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        os.makedirs(logdir, exist_ok=False)
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env
    params['n_episodes'] = kwargs['n_episodes']
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**kwargs)
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    pid = os.getpid()
    logger.info("Starting process id: {}".format(pid))
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running ' + kwargs['algorithm'] +' with just a single MPI worker. This will work, but the HER ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()
    dims = config.configure_dims(params)
    if restore_policy is None:
        policy = config.configure_policy(dims=dims, params=params)
    else:
        policy = config.load_policy(restore_policy_file=restore_policy,  params=params)
        loaded_env_name = policy.info['env_name']
        assert loaded_env_name == env

    # Rollout and evaluation parameters
    rollout_params = config.ROLLOUT_PARAMS
    rollout_params.update(kwargs)

    eval_params = config.EVAL_PARAMS
    eval_params.update(kwargs)

    for name in config.ROLLOUT_PARAMS_LIST:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)
    eval_params['training_rollout_worker'] = rollout_worker
    eval_params['exploit'] = True

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed)

    # early_stop_success_rate = kwargs['early_stop_success_rate'] / 100
    # if 'early_stop_data_column' not in kwargs.keys():
    #     kwargs['early_stop_data_column'] = None
    #     kwargs['early_stop_threshold'] = None
    train(
        logdir=logdir, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_episodes=params['n_episodes'], n_train_batches=params['n_train_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        early_stop_data_column=kwargs['early_stop_data_column'], early_stop_threshold=kwargs['early_stop_threshold']
    )
    # print("Done training for process with rank {}".format(rank))

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@main_linker.click_main
@click.pass_context
def main(ctx, **kwargs):
    global config, RolloutWorker, policy_linker, num_cpu
    config, RolloutWorker = main_linker.import_creator(kwargs['algorithm'])
    policy_args = ctx.forward(main_linker.get_policy_click)
    cmd_line_update_args = {ctx.args[i][2:]: type(policy_args[ctx.args[i][2:]])(ctx.args[i + 1]) for i in
                            range(0, len(ctx.args), 2)}
    policy_args.update(cmd_line_update_args)
    kwargs.update(policy_args)
    num_cpu = kwargs['num_cpu']
    override_params = config.OVERRIDE_PARAMS_LIST
    kwargs['override_params'] = {}
    for op in override_params:
        if op in kwargs.keys():
            kwargs['override_params'][op] = kwargs[op]
    subdir_exists = True

    print("Running training for {}".format(kwargs))
    ctr = kwargs['try_start_idx']
    max_ctr = kwargs['max_try_idx']
    path_params = kwargs['override_params']
    alg_str = kwargs['algorithm'].split(".")[1]
    path_params.update({'info': kwargs['info'], 'alg': alg_str})
    git_label = get_git_label()
    while subdir_exists:
        param_subdir = get_subdir_by_params(path_params, ctr)
        if git_label != '':
            logdir = os.path.join(kwargs['base_logdir'], git_label, kwargs['env'], param_subdir)
        else:
            logdir = os.path.join(kwargs['base_logdir'], kwargs['env'], param_subdir)
        subdir_exists = os.path.exists(logdir)
        ctr += 1

    kwargs['logdir'] = logdir
    kwargs['seed'] = int(time.time())

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
            # Check if training is necessary. It is not if the last run for this configuration did not achieve at least X% success rate.
            min_succ_rate = 0.08
            pass
            # last_res = load_results(last_res_file)
            # if len(last_res['test/success_rate']) == kwargs['n_epochs']:
            #     last_succ_rates = np.mean(last_res['test/success_rate'][-4:])
            #     if last_succ_rates < min_succ_rate:
            #         do_train = False
            #         print("Last time this did only achieve {} success rate avg over last 4 epochs... skipping a new test!".format(
            #             last_succ_rates))
        except:
            print("Could not load progress data {}".format(last_res_file))
    if trial_no > max_ctr:
        print("Already collected enough data for this parameterization")
        do_train = False
    if do_train:
        print("Launching training")
        launch(**kwargs)


if __name__ == '__main__':
    main()