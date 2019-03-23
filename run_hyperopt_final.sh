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


for i in 1 2 3 4 5
do
    penalty_magnitude='20'
    test_subgoal_perc='1'
    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --penalty_magnitude ${penalty_magnitude}
--test_subgoal_perc ${test_subgoal_perc} --info subtractpenalty"
    echo ${cmd}
    ${cmd}

    penalty_magnitude='1'
    test_subgoal_perc='0'
    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --penalty_magnitude ${penalty_magnitude}
--test_subgoal_perc ${test_subgoal_perc} --info subtractpenalty"
    echo ${cmd}
    ${cmd}

penalty_magnitude='20'
    test_subgoal_perc='0.3'
    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
--rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
--base_logdir /data/$(whoami)/herhrl --render 0 --penalty_magnitude ${penalty_magnitude}
--test_subgoal_perc ${test_subgoal_perc} --info subtractpenalty"
    echo ${cmd}
    ${cmd}

done


