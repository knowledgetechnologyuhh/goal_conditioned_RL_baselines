#!/usr/bin/env bash
source ./set_paths.sh
#export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cudnn/lib64:/informatik3/wtm/home/eppe/.mujoco/mjpro150/bin:/usr/lib/nvidia-387
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so

n_cpu=8
n_instances=3
#n_instances=1
bind_core=0

rollout_batch_size=1
n_episodes=100
#n_episodes=10
n_epochs=200
#n_epochs=2
penalty_magnitude=-2
test_subgoal_perc=1
early_stop_threshold=90
n_train_batches=15
min_th=1

n_objects=1
n_subgoals_layers=$(( n_objects*10 ))
max_th=${n_objects}
env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

krenew -K 60 -b

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
        --early_stop_success_rate ${early_stop_threshold}
        "

#for i in 1
for i in 1 2 3
do
    for obs_noise_coeff in '0.0' '0.005' '0.01'
#    for obs_noise_coeff in '0.01'
    do
        noise_cmd="${cmd} --obs_noise_coeff ${obs_noise_coeff}"
        shallow_noise_cmd="${noise_cmd} --policies_layers [] --n_subgoals_layers []"
        subgoal_cmd="${noise_cmd} --n_subgoals_layers [${n_subgoals_layers}]"
        pddl_noise_cmd="${subgoal_cmd} --policies_layers [PDDL_POLICY]"
        mix_noise_cmd="${subgoal_cmd} --policies_layers [MIX_PDDL_HRL_POLICY]"
        hrl_noise_cmd="${subgoal_cmd} --policies_layers [DDPG_HER_HRL_POLICY]"

        for instance in $(seq 1 $n_instances)
        do
            echo ${pddl_noise_cmd}
            ${pddl_noise_cmd} &
            sleep 30
        done
        wait

        for p_steepness in '4.0'
        do
            this_mix_cmd="${mix_noise_cmd} --mix_p_steepness ${p_steepness}"
            for instance in $(seq 1 $n_instances)
            do
                echo ${this_mix_cmd}
                ${this_mix_cmd} &
                sleep 30
            done
            wait
        done

        for instance in $(seq 1 $n_instances)
        do
            echo ${hrl_noise_cmd}
            ${hrl_noise_cmd} &
            sleep 30
        done
        wait

        for instance in $(seq 1 $n_instances)
        do
            echo ${shallow_noise_cmd}
            ${shallow_noise_cmd} &
            sleep 30
        done
        wait
     done
done


