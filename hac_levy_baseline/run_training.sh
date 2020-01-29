#!/bin/bash
source ../set_paths.sh
source ../venv/bin/activate

#cmd="python initialize_HAC.py --mix_train_test --retrain --env ANT_FOUR_ROOMS_2"
cmd="python initialize_HAC.py --mix_train_test --env ANT_REACHER_2_SMALL_SUBG --Q_values --retrain"

max_active_procs=4
n_runs=15

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