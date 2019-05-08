#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=16
rollout_batch_size=1
n_episodes=100
n_epochs=500
n_objects=2
min_th=1
max_th=1
policy="PDDL_POLICY"
easy=$1


env="HookMujocoEnv-sparse-gripper_above-o${n_objects}-h${min_th}-${max_th}-e${easy}-v1"

cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --policies_layers [${policy}] --n_subgoals_layers [6]
--early_stop_success_rate 90 --obs_noise_coeff 0.005"

for i in 1 2 3 4 5
do
    krenew -K 60 -b
    echo ${cmd}

    ${cmd}
done
