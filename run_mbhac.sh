#!/bin/bash

cmd="python experiment/train.py --algorithm baselines.mbhac"
cmd+=" --policy_save_interval 0 --graph 0 --render 0"
cmd+=" --base_logdir data/ --buffer_size 250"

cmd+=" --env AntReacherEnv-v0"
# cmd+=" --env AntFourRoomsEnv-v0"
# cmd+=" --env BlockStackMujocoEnv-gripper_random-o0-v1"
# cmd+=" --env CausalDependenciesMujocoEnv-o0-v0"
# cmd+=" --env UR5ReacherEnv-v1"
# cmd+=" --env CausalDependenciesMujocoEnv-o1-v0"
# cmd+=" --env BlockStackMujocoEnv-gripper_none-o1-v1"
# cmd+=" --env HookMujocoEnv-gripper_random-e1-v1"
# cmd+=" --env BlockStackMujocoEnv-gripper_none-o2-v1"

cmd+=" --n_epochs 20 --n_train_rollouts 100 --n_test_rollouts 25"
cmd+=" --model_based 1 --mb_hidden_size 128,128,128 --eta 0.75"

max_active_procs=8
n_runs=16

cmd_ctr=0
until [ $cmd_ctr -gt $n_runs ]
do
  ((cmd_ctr++))
  echo "Next cmd in queue is:"
  echo $cmd
  n_active_procs=$(pgrep -c -P$$)
  ps -ef | grep sleep
  echo "Currently, there are ${n_active_procs} active processes"
  while [ "$n_active_procs" -ge "$max_active_procs" ];do
      echo "${n_active_procs} processes running; queue is full, waiting..."
      sleep 15
      n_active_procs=$(pgrep -c -P$$)
  done
  echo "Now executing cmd ${cmd_ctr}: "
  echo ${cmd}
  $cmd || true & # Execute in background
  sleep 15
done
echo "All commands have been executed"
