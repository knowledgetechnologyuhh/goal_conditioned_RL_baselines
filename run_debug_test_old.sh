#!/bin/bash
source ./set_paths.sh
n_train_rollouts=3
n_test_rollouts=2
n_epochs=2
rollout_batch_size=1
n_cpu=2
penalty_magnitude=10

environments=( 'BlocksMujocoEnv-gripper_random-o2-v1' 'BlocksMujocoEnv-gripper_above-o1-v1' 'AntFourRoomsEnv-v0' )
policies=( 'PDDL_POLICY' 'DDPG_HER_HRL_POLICY' 'DDPG_HER_HRL_POLICY_SHARED_PREPROC' )
declare -a two_layer_policy_combinations=()

two_layer_policy_combinations+=( "${policy_comb}" )

policy_combinations=( '[PDDL_POLICY,DDPG_HER_HRL_POLICY]' '[DDPG_HER_HRL_POLICY,DDPG_HER_HRL_POLICY]' '[DDPG_HER_HRL_POLICY_SHARED_PREPROC,DDPG_HER_HRL_POLICY_SHARED_PREPROC]' '[PDDL_POLICY,DDPG_HER_HRL_POLICY_SHARED_PREPROC]' )
network_classes=( 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttnReduced' 'baselines.herhrl.actor_critic:ActorCriticProbSampling' 'baselines.herhrl.actor_critic:ActorCriticProbSamplingReduced' )
action_steps_combinations=( '[4,20]' )

shared_preproc_network_classes=( 'actor_critic:ActorCritic' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttnHeaviside' 'actor_critic_shared_preproc:ActorCriticVanillaAttnEnforceW' 'actor_critic_shared_preproc:ActorCriticProbSamplingAttn' )


#environments=( 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1' 'AntFourRoomsEnv-v0' )
#policy_combinations=( '[DDPG_HER_HRL_POLICY_SHARED_LOSS,DDPG_HER_HRL_POLICY_SHARED_LOSS]' )
#network_classes=( 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' )


base_cmd="python3 experiment/train.py
                  --num_cpu ${n_cpu}
                  --rollout_batch_size ${rollout_batch_size}
                  --n_epochs ${n_epochs}
                  --n_train_rollouts ${n_train_rollouts}
                  --base_logdir /data/$(whoami)/herhrl
                  --render 0
                  --n_test_rollouts ${n_test_rollouts}
                  --early_stop_success_rate 99"

# Iterate through environments
for env in "${environments[@]}"
do
  env_cmd="${base_cmd} --env ${env}"

  # Iterate through network classes
  for network_class in "${network_classes[@]}"
  do
    net_cmd="${env_cmd} --network_class ${network_class}"

    # Generate and execute HER command
    her_cmd="${net_cmd} --algorithm baselines.her"
    echo ${her_cmd}
    ${her_cmd}

    # Generate and execute HERHRL commands
    herhrl_cmd="${net_cmd} --algorithm baselines.herhrl"

    # Iterate through action_steps
    for action_steps in "${action_steps_combinations[@]}"
    do
      action_steps_cmd="${herhrl_cmd} --action_steps ${action_steps}"

      # Iterate through policy_combinations
      for policy_combination in "${policy_combinations[@]}"
      do
        policy_comb_cmd="${action_steps_cmd} --policies_layers ${policy_combination}"
        echo ${policy_comb_cmd}
        ${policy_comb_cmd}
      done
    done
  done
done
