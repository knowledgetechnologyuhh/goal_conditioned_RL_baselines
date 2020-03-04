#!/bin/bash

cmd="python experiment/train.py --algorithm baselines.hac --num_cpu 1 --render 0 --n_epochs 100  --n_train_rollouts 100 --n_test_rollouts 30"

max_active_procs=16
n_runs=8

# declare -a repeated_cmd_array=()
cmd_ctr=0
for (( j=0; j<n_runs; ++j))
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
