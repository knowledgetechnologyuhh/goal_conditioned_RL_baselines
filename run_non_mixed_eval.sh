#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=16
rollout_batch_size=2
n_episodes=50
n_epochs=200

#n_episodes=5
#n_epochs=2

n_objects=1
min_th=1
max_th=1


krenew -K 60 -b
env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

#cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
#        --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
#        --base_logdir /data/$(whoami)/herhrl --render 0 --policies_layers [] --n_subgoals_layers []"
#            echo ${cmd}
#            ${cmd}

for i in 1 2 3 4 5
do
    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
        --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
        --base_logdir /data/$(whoami)/herhrl --render 0 --policies_layers [] --n_subgoals_layers []"
            echo ${cmd}
            ${cmd}

    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
        --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
        --base_logdir /data/$(whoami)/herhrl --render 0 --policies_layers [PDDL_POLICY] --n_subgoals_layers [3]"
            echo ${cmd}
            ${cmd}

    cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl
        --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes}
        --base_logdir /data/$(whoami)/herhrl --render 0 --policies_layers [DDPG_HER_HRL_POLICY] --n_subgoals_layers [3]"
            echo ${cmd}
            ${cmd}
done


