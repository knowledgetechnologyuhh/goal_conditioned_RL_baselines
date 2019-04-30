#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=8
rollout_batch_size=1
n_episodes=50
n_epochs=500
n_objects=2
min_th=1
max_th=1
policy="PDDL_POLICY"


krenew -K 60 -b
env="HookMujocoEnv-sparse-gripper_above-o${n_objects}-h${min_th}-${max_th}-v1"

cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --policies_layers [${policy}] --n_subgoals_layers [5]"

for i in 1 2 3 4 5
do
    echo ${cmd}

    ${cmd}
done
