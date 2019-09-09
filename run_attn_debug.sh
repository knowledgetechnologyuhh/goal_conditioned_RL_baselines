#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=6
n_epochs=100
#early_stop_threshold=70
initial_trial_idx=100
#env="AntFourRoomsEnv-v0"
max_active_procs=5
max_trials_per_config=5
early_stop_threshold='9.9'
early_stop_value='test/subgoals_achieved'
total_cmd_ctr=0

krenew -K 60 -b
declare -a cmd_array=()
end_trial_idx=$(( $initial_trial_idx + $max_trials_per_config ))
#--base_logdir /storage/wtmgws/wtmgws7_data/ideas_deep_rl/data
#for network_class in 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttnReduced' 'baselines.herhrl.actor_critic:ActorCriticProbSampling' 'baselines.herhrl.actor_critic:ActorCriticProbSamplingReduced'
#for network_class in 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttnReduced'
for network_class in 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttnReduced'
do
  for l2_action in '1.0' '0.0'
  do
    for shared_pi_err_coeff in '0.0' '1.0' '0.2'
    do
      if [[ ( $shared_pi_err_coeff != '0.0' ) && ( $network_class = 'baselines.herhrl.actor_critic:ActorCritic' )  ]]; then
        continue
      fi
      for env in 'AntFourRoomsEnv-v0' 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1' 'TowerBuildMujocoEnv-sparse-gripper_above-o2-h1-2-v1'
      do
        cmd="python3 experiment/train.py
        --early_stop_threshold ${early_stop_threshold}
        --early_stop_data_column ${early_stop_value}
        --action_steps [10,25]
        --policies_layers [DDPG_HER_HRL_POLICY_SHARED_LOSS,DDPG_HER_HRL_POLICY_SHARED_LOSS]
        --n_episodes 100
        --n_test_rollouts 10
        --n_train_batches 15
        --env ${env}
        --algorithm baselines.herhrl
        --render 0
        --num_cpu ${n_cpu}
        --penalty_magnitude 10
        --n_epochs 100
        --try_start_idx ${initial_trial_idx}
        --max_try_idx ${end_trial_idx}
        --base_logdir /data/$(whoami)/herhrl
        --network_class ${network_class}
        --shared_pi_err_coeff ${shared_pi_err_coeff}
        --action_l2 ${l2_action}"
        echo ${cmd}
    #    cmd="sleep 7"
        ((total_cmd_ctr++))
        cmd_array+=( "${cmd}" )
      done
    done
  done
done

echo "Total number of commands: ${total_cmd_ctr}"

declare -a repeated_cmd_array=()
for (( i=0; i<$max_trials_per_config; ++i))
do
  repeated_cmd_array+=( "${cmd_array[@]}" )
done

for cmd in "${repeated_cmd_array[@]}"
do
    echo "Next cmd in queue is ${cmd}"
    n_active_procs=$(pgrep -c -P$$)
    echo "Currently, there are ${n_active_procs} active processes"
    while [ "$n_active_procs" -ge "$max_active_procs" ];do
        echo "${n_active_procs} processes running; queue is full, waiting..."
        sleep 60
        n_active_procs=$(pgrep -c -P$$)
    done
    echo "Now executing ${cmd}"
    ${cmd} || true &
    sleep 30
done





