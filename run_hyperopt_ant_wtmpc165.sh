#!/usr/bin/env bash
#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cudnn/lib64:/informatik3/wtm/home/eppe/.mujoco/mjpro150/bin:/usr/lib/nvidia-387
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so

n_cpu=1
n_instances=4
bind_core=0

early_stop_threshold=99

#min_th=1
#--n_subgoals_layers
#[12,700]
#--policies_layers
#[PDDL_POLICY,DDPG_HER_HRL_POLICY]
#--n_episodes
#10
#--n_test_rollouts
#2
#--env
#AntFourRoomsEnv-v0
#--algorithm
#baselines.herhrl
#--render
#1
#--num_cpu
#1
#--penalty_magnitude
#1
#--n_epochs
#100
#--network_class
#baselines.herhrl.actor_critic:ActorCritic

env="AntFourRoomsEnv-v0"

krenew -K 60 -b

n_subgoals_layers='[15,700]'

for n_episodes in 20 40 80
do
    n_epochs=$(( 4000 / $n_episodes ))
    cmd="python3 experiment/train.py
        --num_cpu ${n_cpu}
        --env ${env}
        --algorithm baselines.herhrl
        --n_epochs ${n_epochs}
        --n_episodes ${n_episodes}
        --base_logdir /data/$(whoami)/herhrl
        --render 0
        --early_stop_success_rate ${early_stop_threshold}
        --info nepi${n_episodes}
        "
    for i in 1 2
    do
        for obs_noise_coeff in '0.0'
        do
            noise_cmd="${cmd} --obs_noise_coeff ${obs_noise_coeff}"
            shallow_noise_cmd="${noise_cmd} --policies_layers [] --n_subgoals_layers []"
            subgoal_cmd="${noise_cmd} --n_subgoals_layers ${n_subgoals_layers}"
            pddl_noise_cmd="${subgoal_cmd} --policies_layers [PDDL_POLICY,DDPG_HER_HRL_POLICY]"
#            mix_noise_cmd="${subgoal_cmd} --policies_layers [MIX_PDDL_HRL_POLICY]"
#            hrl_noise_cmd="${subgoal_cmd} --policies_layers [DDPG_HER_HRL_POLICY]"

            for instance in $(seq 1 $n_instances)
            do
                echo ${pddl_noise_cmd}
                ${pddl_noise_cmd} &
                sleep 30
            done
            wait

         done
    done
done



