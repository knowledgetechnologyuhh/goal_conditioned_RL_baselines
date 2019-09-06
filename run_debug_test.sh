#!/bin/bash
source ./set_paths.sh
n_episodes=3
n_test_rollouts=2
n_epochs=2
rollout_batch_size=1
n_cpu=2
penalty_magnitude=10
environments=( 'TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-2-v1' 'TowerBuildMujocoEnv-sparse-gripper_above-o1-h1-1-v1' 'AntFourRoomsEnv-v0' )
policy_combinations=( '[PDDL_POLICY,DDPG_HER_HRL_POLICY]' '[DDPG_HER_HRL_POLICY,DDPG_HER_HRL_POLICY]' )
network_classes=( 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttnReduced' 'baselines.herhrl.actor_critic:ActorCriticProbSampling' 'baselines.herhrl.actor_critic:ActorCriticProbSamplingReduced' )
action_steps_combinations=( '[4,20]' )

base_cmd="python3 experiment/train.py
                  --num_cpu ${n_cpu}
                  --rollout_batch_size ${rollout_batch_size}
                  --n_epochs ${n_epochs}
                  --n_episodes ${n_episodes}
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
