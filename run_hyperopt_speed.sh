#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=14
rollout_batch_size=1
#n_episodes=100
n_epochs=500
penalty_magnitude=-2
test_subgoal_perc=1
early_stop_threshold=80
#n_train_batches=20
min_th=1

n_instances=1

krenew -K 60 -b

for n_objects in 1 2
do
    n_subgoals_layers=$(( n_objects*6 ))
    max_th=$n_objects
    for n_episodes in 100 150 200 75
    do
        for n_train_batches in 20 10 30 15 25
        do
            for i in 1 2
            do

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
                    --early_stop_success_rate ${early_stop_threshold}"
                echo ${cmd}
                for instance in $(seq 1 $n_instances)
                do
                    ${cmd} &
                    sleep 30
                done
                wait

             done
        done
    done
done


