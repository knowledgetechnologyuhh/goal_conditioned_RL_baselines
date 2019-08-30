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
action_steps_combinations=( '[4,20]' )

for env in "${environments[@]}"
do
  for action_steps in "${action_steps_combinations[@]}"
  do
    for policy_combination in "${policy_combinations[@]}"
    do
                cmd="python3 experiment/train.py
                --num_cpu ${n_cpu}
                --env ${env}
                --algorithm baselines.herhrl
                --rollout_batch_size ${rollout_batch_size}
                --n_epochs ${n_epochs}
                --n_episodes ${n_episodes}
                --base_logdir /data/$(whoami)/herhrl
                --render 0
                --n_test_rollouts ${n_test_rollouts}
                --penalty_magnitude ${penalty_magnitude}
                --policies_layers ${policy_combination}
                --action_steps ${action_steps}
                --early_stop_success_rate 99"
                echo ${cmd}
                ${cmd}
    done
  done
done


#python3 experiment/train.py --n_episodes ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [7] --policies_layers [DDPG_HER_HRL_POLICY]
#
#python3 experiment/train.py --n_episodes ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [7] --policies_layers [PDDL_POLICY]
#
#python3 experiment/train.py --n_episodes ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [7] --policies_layers [MIX_PDDL_HRL_POLICY]
#
#python3 experiment/train.py --n_episodes ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.her_pddl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs}
#
#python3 experiment/train.py --n_episodes ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.her --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs}
#
#
