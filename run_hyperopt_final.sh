#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=16
rollout_batch_size=2
n_episodes=50
n_epochs=1000
n_objects=1
min_th=1
max_th=1
penalty_magnitude=1
test_subgoal_perc=0

krenew -K 60 -b
env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"


for i in 1 2 3
do
    p_threshold='0.2'
    p_steepness='2.0'
    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --penalty_magnitude ${penalty_magnitude}
--test_subgoal_perc ${test_subgoal_perc} --policies_layers [MIX_PDDL_HRL_POLICY] --mix_p_threshold ${p_threshold}
--mix_p_steepness ${p_steepness}"
    echo ${cmd}
    ${cmd}

done


