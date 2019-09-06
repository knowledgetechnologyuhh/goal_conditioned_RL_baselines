#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=6
n_epochs=70
early_stop_threshold=70
initial_trial_idx=100
env="AntFourRoomsEnv-v0"
max_active_procs=1
max_trials_per_config=3

krenew -K 60 -b
declare -a cmd_array=()
end_trial_idx=$(( $initial_trial_idx + $max_trials_per_config ))
#--base_logdir /storage/wtmgws/wtmgws7_data/ideas_deep_rl/data
for network_class in 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttnReduced' 'baselines.herhrl.actor_critic:ActorCriticProbSampling' 'baselines.herhrl.actor_critic:ActorCriticProbSamplingReduced'
do
    for policies_layers in '[DDPG_HER_HRL_POLICY_SHARED_LOSS_PI,DDPG_HER_HRL_POLICY_SHARED_LOSS_PI]' '[DDPG_HER_HRL_POLICY,DDPG_HER_HRL_POLICY]' '[DDPG_HER_HRL_POLICY,DDPG_HER_HRL_POLICY_SHARED_LOSS_PI]'
    do
        cmd="python3 experiment/train.py
        --action_steps [10,25]
        --policies_layers ${policies_layers}
        --n_episodes 100
        --n_test_rollouts 10
        --n_train_batches 15
        --env ${env}
        --algorithm baselines.herhrl
        --render 0
        --num_cpu 4
        --penalty_magnitude 10
        --n_epochs 100
        --try_start_idx=${initial_trial_idx}
        --max_try_idx=${end_trial_idx}
        --network_class ${network_class}"
        echo ${cmd}
    #    cmd="sleep 7"
        cmd_array+=( "${cmd}" )
    done
done

declare -a repeated_cmd_array=()
for (( i=0; i<$max_trials_per_config; ++i))
do
  repeated_cmd_array+=( "${cmd_array[@]}" )
  echo $i
done

for cmd in "${repeated_cmd_array[@]}"
do
    echo "Next cmd in queue is ${cmd}"
    n_active_procs=$(pgrep -c -P$$)
    echo "Currently, there are ${n_active_procs} active processes"
    while [ "$n_active_procs" -ge "$max_active_procs" ];do
        echo "Process queue is full, waiting..."
        sleep 120
        n_active_procs=$(pgrep -c -P$$)
    done
    echo "Now executing ${cmd}"
    ${cmd} || true &
    sleep 30
done





