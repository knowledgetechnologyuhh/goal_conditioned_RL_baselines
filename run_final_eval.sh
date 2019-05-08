#!/usr/bin/env bash
source ./set_paths.sh
#export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cudnn/lib64:/informatik3/wtm/home/eppe/.mujoco/mjpro150/bin:/usr/lib/nvidia-387
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so

n_cpu=8
n_instances=1
bind_core=0

n_epochs=500
early_stop_threshold=90

min_th=1
n_objects=1
n_subgoals_layers=$(( n_objects*6 ))
max_th=${n_objects}
env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

krenew -K 60 -b

cmd="python3 experiment/train.py
        --num_cpu ${n_cpu}
        --env ${env}
        --algorithm baselines.herhrl
        --n_epochs ${n_epochs}
        --base_logdir /data/$(whoami)/herhrl
        --render 0
        --early_stop_success_rate ${early_stop_threshold}
        "

for i in 1 2 3
do
    for obs_noise_coeff in '0.0' '0.005' '0.01'
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

     done
done


