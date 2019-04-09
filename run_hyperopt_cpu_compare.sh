#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=8
rollout_batch_size=1
n_episodes=100
n_epochs=500
penalty_magnitude=-2
test_subgoal_perc=1
early_stop_threshold=80
n_train_batches=20
min_th=1

krenew -K 60 -b

for i in 1 2
do
    for n_objects in 1
    do
        n_subgoals_layers=$(( n_objects*6 ))
        max_th=$n_objects
        env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

        cmd="python3 experiment/train.py
                --num_cpu ${n_cpu}
                --env ${env}
                --algorithm baselines.herhrl
            --rollout_batch_size ${rollout_batch_size}
            --n_epochs ${n_epochs}
            --n_episodes ${n_episodes}
            --n_train_batches ${n_train_batches}
            --base_logdir /data/$(whoami)/herhrl
            --render 0
            --penalty_magnitude ${penalty_magnitude}
            --test_subgoal_perc ${test_subgoal_perc}
            --policies_layers [PDDL_POLICY]
            --n_subgoals_layers [${n_subgoals_layers}]
            --info cpu${n_cpu}"
                echo ${cmd}
                ${cmd}
    done
done


