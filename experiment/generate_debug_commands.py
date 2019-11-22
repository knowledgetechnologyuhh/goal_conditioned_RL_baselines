import os
import getpass

def get_her_cmds(base_cmd):
    cmd = base_cmd + " --algorithm baselines.her"
    return [cmd]

def get_herhrl_cmds(base_cmd):
    base_cmd += " --action_steps [4,20]"
    base_cmd += " --algorithm baselines.herhrl"
    hl_policies = ['PDDL_POLICY', 'DDPG_HER_HRL_POLICY']
    ll_policies = ['DDPG_HER_HRL_POLICY']
    policy_acs = {}
    policy_acs["PDDL_POLICY"] = ["None"]
    policy_acs["DDPG_HER_HRL_POLICY"] = ['actor_critic:ActorCritic','actor_critic:ActorCriticSharedPreproc','actor_critic:ActorCriticProbSamplingAttn']
    policy_combinations = []
    ac_combinations = []
    for hlp in hl_policies:
        for llp in ll_policies:
            for hlpac in policy_acs[hlp]:
                for llpac in policy_acs[llp]:
                    policy_combinations.append("[{},{}]".format(hlp, llp))
                    ac_combinations.append('[{},{}]'.format(hlpac, llpac))

    all_cmds = []
    for polcomb, accomb in zip(policy_combinations, ac_combinations):
        this_cmd = base_cmd
        this_cmd += " --policies_layers {}".format(polcomb)
        this_cmd += " --network_classes {}".format(accomb)
        all_cmds.append(this_cmd)

    return all_cmds

if __name__ == "__main__":
    cmds = []

    n_train_rollouts = 2
    n_test_rollouts = 2
    n_epochs = 2
    rollout_batch_size = 1
    n_cpu = 2
    penalty_magnitude = 10

    environments = ['BlocksMujocoEnv-gripper_random-o2-v1',
                    'BlocksMujocoEnv-gripper_above-o1-v1',
                    'AntFourRoomsEnv-v0',
                    'HookMujocoEnv-gripper_above-e1-v1']

    whoami = getpass.getuser()
    opts_values = {"general": {}}
    opts_values["general"]['num_cpu'] = n_cpu
    opts_values["general"]['rollout_batch_size'] = rollout_batch_size
    opts_values["general"]['n_epochs'] = n_epochs
    opts_values["general"]['n_train_rollouts'] = n_train_rollouts
    opts_values["general"]['n_test_rollouts'] = n_test_rollouts
    opts_values["general"]['base_logdir'] = "/data/" + whoami + "/herhrl"
    opts_values["general"]['render'] = 0
    opts_values["general"]['try_start_idx'] = 100
    opts_values["general"]['max_try_idx'] = 500

    base_cmd = "python3 experiment/train.py"
    for k, v in sorted(opts_values["general"].items()):
        base_cmd += " --{}".format(k) + " {}".format(str(v))

    algorithms = ["baselines.her", "baselines.herhrl"]
    get_cmd_functions = {}
    get_cmd_functions["baselines.her"] = get_her_cmds
    get_cmd_functions["baselines.herhrl"] = get_herhrl_cmds

    for env in environments:
        env_base_cmd = base_cmd + " --env {}".format(env)
        for alg in algorithms:
            cmds += get_cmd_functions[alg](env_base_cmd)

    cmd_file_name = "debug_cmds.txt"
    with open(cmd_file_name, "w") as cmd_file:
        cmd_file.write("")
    with open(cmd_file_name, "a") as cmd_file:
        for cmd in cmds:
           cmd_file.write(cmd +"\n")
    print("Done generating debug commands")