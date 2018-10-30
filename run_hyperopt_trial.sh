#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cudnn/lib64:/informatik3/wtm/home/eppe/.mujoco/mjpro150/bin:/usr/lib/nvidia-387
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so


n_cpu=1
rollout_batch_size=2
n_episodes=2
n_epochs=50
n_objects=4
min_th=1
max_th=1
buff_sampling='random'
memval_method='uniform'
action_selection='random'


n_train_batches=$1
model_lr=$2
action_selection=$3
memval_method=$4
buff_sampling=$5


env="BuildTowerMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.model_based --rollout_batch_size ${rollout_batch_size} --n_train_batches ${n_train_batches} --n_epochs ${n_epochs} --n_episodes ${n_episodes} --base_logdir /data/eppe/baselines_data --render 0 --buff_sampling ${buff_sampling} --memval_method ${memval_method} --action_selection ${action_selection}"

echo ${cmd}

${cmd}



