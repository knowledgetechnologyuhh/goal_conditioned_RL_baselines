#!/usr/bin/env bash
source ./set_paths.sh

python3 experiment/generate_debug_commands.py
sleep 2
cmd_file="debug_cmds.txt"
max_active_procs=8
cmd_ctr=0
n_cmds=$(cat $cmd_file | wc -l)
while IFS= read -r cmd
do
    ((cmd_ctr++))
    echo "Next cmd in queue is ${cmd}"
#    cmd='sleep 12' # Uncomment for debugging this script with a simple sleep command
    n_active_procs=$(pgrep -c -P$$)
    ps -ef | grep sleep
    echo "Currently, there are ${n_active_procs} active processes"
    while [ "$n_active_procs" -ge "$max_active_procs" ];do
        echo "${n_active_procs} processes running; queue is full, waiting..."
        sleep 5
        n_active_procs=$(pgrep -c -P$$)
    done
    echo "Now executing cmd ${cmd_ctr} / ${n_cmds}: "
    echo ${cmd}
    ${cmd} || true & # Execute in background
    sleep 1
done < $cmd_file
echo "All commands have been ececuted"
