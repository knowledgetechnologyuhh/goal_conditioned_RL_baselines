#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=6
#n_epochs=75
#early_stop_threshold=70
initial_trial_idx=100
n_test_rollouts=10
n_episodes=100
#env="AntFourRoomsEnv-v0"
max_active_procs=5
max_trials_per_config=2
early_stop_threshold='9.9'
early_stop_value='test/subgoals_achieved'
total_cmd_ctr=0

krenew -K 60 -b
declare -a cmd_array=()
end_trial_idx=$(( $initial_trial_idx + $max_trials_per_config - 1 ))
#for ll_network_class in 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnSteep6' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnSteep100' 'actor_critic_shared_preproc:ActorCritic' 'actor_critic_shared_preproc:ActorCriticSharedPreproc' 'actor_critic_shared_preproc:ActorCriticVanillaAttn' 'actor_critic_shared_preproc:ActorCriticVanillaAttnReduced'
#for ll_network_class in 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnSteep6' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnSteep100' 'actor_critic_shared_preproc:ActorCriticVanillaAttnSteep6' 'actor_critic_shared_preproc:ActorCriticVanillaAttnSteep100'
for ll_network_class in 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnSteep6' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnSteep100' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnTwoValSteep6' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnTwoValSteep100'
do
  network_classes="[actor_critic:ActorCritic,${ll_network_class}]"
  for l2_action in '1.0'
  do
    for shared_pi_err_coeff in '0.2'
    do
      if [[ ( $shared_pi_err_coeff != '0.0' ) && ( $ll_network_class = 'actor_critic_shared_preproc:ActorCritic' )  ]]; then
        continue
      fi
#      for env in 'AntFourRoomsEnv-v0' 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1' 'TowerBuildMujocoEnv-sparse-gripper_above-o2-h1-2-v1'
#      for env in 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1' 'TowerBuildMujocoEnv-sparse-gripper_above-o2-h1-2-v1'
      for env in 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1'
      do
        if [[ ( $env = 'AntFourRoomsEnv-v0' ) || ( $env = 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1' )  ]]; then
          n_epochs=40
        else
          n_epochs=100
        fi
#        # For debugging uncomment those:
        n_epochs=3
        n_episodes=3
        n_test_rollouts=2

        cmd="python3 experiment/train.py
        --early_stop_threshold ${early_stop_threshold}
        --early_stop_data_column ${early_stop_value}
        --action_steps [10,25]
        --policies_layers [DDPG_HER_HRL_POLICY,DDPG_HER_HRL_POLICY_SHARED_PREPROC]
        --n_episodes ${n_episodes}
        --n_test_rollouts ${n_test_rollouts}
        --n_train_batches 15
        --env ${env}
        --algorithm baselines.herhrl
        --render 0
        --num_cpu ${n_cpu}
        --penalty_magnitude 10
        --n_epochs ${n_epochs}
        --try_start_idx ${initial_trial_idx}
        --max_try_idx ${end_trial_idx}
        --base_logdir /data/$(whoami)/herhrl
        --network_classes ${network_classes}
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
#    ${cmd} # For debugging: execute in foreground
    ${cmd} || true & # Execute in background
    sleep 30
done





