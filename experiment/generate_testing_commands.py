import getpass

"""
This script is used for testing backwards compatibility after adding a new feature.
If you want to merge your development branch with the overall devel branch, please proceed as described in README.md file
"""


class TestingConfig:
    algorithms = ['herhrl', 'her', 'chac']
    environments = [
        'BlockStackMujocoEnv-gripper_random-o0-v1',
        'BlockStackMujocoEnv-gripper_random-o2-v1',
        'BlockStackMujocoEnv-gripper_above-o1-v1',
        'AntFourRoomsEnv-v0', 'AntReacherEnv-v0',
        'HookMujocoEnv-gripper_above-e1-v1',
        'CausalDependenciesMujocoEnv-o1-v0',
        'CopReacherEnv-ik1-v0',
        'CopReacherEnv-ik0-v0',
        'UR5ReacherEnv-v1']


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
    policy_acs["DDPG_HER_HRL_POLICY"] = ['actor_critic:ActorCritic']
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

def get_chac_cmds(base_cmd):
    base_cmd = base_cmd + " --algorithm baselines.chac"
    forward_model = [0, 1]
    forward_model_hs = ['32', '32,32', '32,32,32']
    n_levels = [1, 2]
    time_scales = ['10', '10,3']

    all_cmds = []
    for fw in forward_model:
        for levels, tscale in zip(n_levels, time_scales):
            cmd = base_cmd
            cmd += " --fw {}".format(fw)
            cmd += " --n_levels {}".format(levels)
            cmd += " --time_scales {}".format(fw)
            if fw:
                for fwhs in forward_model_hs:
                    cmds.append(cmd + " --fw_hidden_size {}".format(fwhs))
            else:
                cmds.append(cmd)

    return all_cmds

if __name__ == "__main__":
    cmds = []

    n_train_rollouts = 2
    n_test_rollouts = 2
    n_epochs = 2
    rollout_batch_size = 1
    opts_values = {"general": {}}

    whoami = getpass.getuser()
    opts_values["general"]['base_logdir'] = "/data/" + whoami + "/herhrl"
    opts_values["general"]['try_start_idx'] = 100
    opts_values["general"]['max_try_idx'] = 500
    opts_values["general"]['render'] = 0
    opts_values["general"]['num_cpu'] = 1
    opts_values["general"]['rollout_batch_size'] = 1
    opts_values["general"]['n_epochs'] = 2
    opts_values["general"]['n_train_rollouts'] = 2
    opts_values["general"]['n_test_rollouts'] = 2

    base_cmd = "python3 experiment/train.py"
    for k, v in sorted(opts_values["general"].items()):
        base_cmd += " --{}".format(k) + " {}".format(str(v))

    get_cmd_functions = {}
    for alg in TestingConfig.algorithms:
        get_cmd_functions["baselines.{}".format(alg)] = eval("get_{}_cmds".format(alg))

    for env in TestingConfig.environments:
        env_base_cmd = base_cmd + " --env {}".format(env)
        for alg in TestingConfig.algorithms:
            cmds += get_cmd_functions["baselines.{}".format(alg)](env_base_cmd)

    cmd_file_name = "test_cmds.txt"
    with open(cmd_file_name, "w") as cmd_file:
        cmd_file.write("")
    with open(cmd_file_name, "a") as cmd_file:
        for cmd in cmds:
            cmd_file.write(cmd + "\n")
    print("Done generating debug commands")