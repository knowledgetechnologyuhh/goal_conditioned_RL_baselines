#!/usr/bin/env bash
source ./set_paths.sh
logs_dir=testing_logs
#rm -rf ${logs_dir}
#rm ${cmd_file}
mkdir ${logs_dir}
python3 experiment/generate_testing_commands.py
sleep 2
cmd_file="test_cmds.txt"
max_active_procs=4
cmd_ctr=0
n_cmds=$(cat $cmd_file | wc -l)
declare -a cmd_arr=()
while IFS= read -r cmd
do
    cmd_arr+=("${cmd[@]}")
done < $cmd_file



for ((i = 0; i < ${#cmd_arr[@]}; i++))
do
#    echo "${cmd_arr[$i]}"
    cmd="${cmd_arr[$i]}"
    ((cmd_ctr++))
    echo "Next cmd in queue is:"
    echo $cmd
#    cmd="sleep 12" # Uncomment for debugging this script with a simple sleep command
    n_active_procs=$(pgrep -c -P$$)
    ps -ef | grep sleep
    echo "Currently, there are ${n_active_procs} active processes"
    while [ "$n_active_procs" -ge "$max_active_procs" ];do
        echo "${n_active_procs} processes running; queue is full, waiting..."
        sleep 15
        n_active_procs=$(pgrep -c -P$$)
    done
    echo "Now executing cmd ${cmd_ctr} / ${n_cmds}: "
    echo ${cmd}
#    ${cmd}
    $cmd 1> ${logs_dir}/${cmd_ctr}.log 2> ${logs_dir}/${cmd_ctr}_err.log || true & # Execute in background
    sleep 3
done
echo "All commands have been executed"


python3 experiment/check_error_logs.py