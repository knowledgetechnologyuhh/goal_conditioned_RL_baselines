#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=16
rollout_batch_size=2
n_episodes=50
n_epochs=150
n_objects=1
min_th=1
max_th=1

krenew -K 60 -b
env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --penalty_magnitude ${penalty_magnitude}
--test_subgoal_perc ${test_subgoal_perc}"


for i in 1 2 3
do
    penalty_magnitude='10'
    test_subgoal_perc='0.1'
    echo ${cmd}
    ${cmd}

    penalty_magnitude='30'
    test_subgoal_perc='0.5'
    echo ${cmd}
    ${cmd}


    penalty_magnitude='20'
    test_subgoal_perc='0.2'
    echo ${cmd}
    ${cmd}

    penalty_magnitude='10'
    test_subgoal_perc='1.0'
    echo ${cmd}
    ${cmd}
done


