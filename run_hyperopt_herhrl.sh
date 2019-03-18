#!/usr/bin/env bash
#export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cudnn/lib64:/informatik3/wtm/home/eppe/.mujoco/mjpro150/bin:/usr/lib/nvidia-387
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
source ./set_paths.sh

n_cpu=4
rollout_batch_size=2
n_episodes=50
n_epochs=150
n_objects=1
min_th=1
max_th=1
#buff_sampling='random'
#memval_method='uniform'
#action_selection='random'



#n_train_batches=$1
#model_lr=$2
#action_selection=$3
#memval_method=$4
#buff_sampling=$5
penalty_magnitude=$1
test_subgoal_perc=$2


env="TowerBuildMujocoEnv-sparse-gripper_random-o${n_objects}-h${min_th}-${max_th}-v1"

cmd="python3 experiment/train.py --num_cpu ${n_cpu} --env ${env} --algorithm baselines.herhrl --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_episodes ${n_episodes} --base_logdir /data/pnguyen/herhrl --render 0 --penalty_magnitude ${penalty_magnitude} --test_subgoal_perc ${test_subgoal_perc}"

echo ${cmd}

${cmd}



