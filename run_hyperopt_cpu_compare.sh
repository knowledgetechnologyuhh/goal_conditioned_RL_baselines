#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=1
rollout_batch_size=1
n_episodes=100
n_epochs=500
penalty_magnitude=-2
test_subgoal_perc=1
early_stop_threshold=80
n_train_batches=20
min_th=1

krenew -K 60 -b

for i in 1 2 3
do
    for n_objects in 1 2
    do
        n_subgoals_layers=$(( n_objects*6 ))
        max_th=$n_objects
        env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"
        for n_cpu in 1 4 8 14
        do
            for n_instances in 1 2 4 8
            do
                for instance in $(seq 1 $n_instances)
                do
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
                    --early_stop_success_rate ${early_stop_threshold}
                        --info cpu${n_instances}x${n_cpu}"
                    echo ${cmd}
                    ${cmd} &
                    sleep 30
                done
                wait
            done
        done
    done
done


