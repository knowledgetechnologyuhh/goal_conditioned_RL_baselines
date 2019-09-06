#!/usr/bin/env bash
source ./set_paths.sh

n_cpu=4
n_epochs=100
early_stop_threshold=95

env="AntFourRoomsEnv-v0"

krenew -K 60 -b

for i in 1 2 3
do
    for network_class in 'baselines.herhrl.actor_critic:ActorCritic' 'baselines.herhrl.actor_critic:ActorCriticSharedPreproc' 'baselines.herhrl.actor_critic:ActorCriticVanillaAttn'
    do
        cmd="python3 experiment/train.py
        --action_steps [5,50]
        --policies_layers [DDPG_HER_HRL_POLICY,DDPG_HER_HRL_POLICY]
        --n_episodes 100
        --n_test_rollouts 10
        --env ${env}
        --algorithm baselines.herhrl
        --render 0
        --num_cpu 4
        --penalty_magnitude 10
        --n_epochs 100
        --network_class ${network_class}"
        echo ${cmd}
        $cmd
     done
done


