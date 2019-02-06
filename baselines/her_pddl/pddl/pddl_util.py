import numpy as np
from baselines.her_pddl.pddl.propositional_planner import Propositional_Planner
import time

def obs_to_preds(obs, goal, n_objects,
                 grasp_xy_threshold=[0.0, 0.025], grasp_z_threshold=[-0.015, 0.02],
                 grip_open_threshold=[0.038, 1.0], grip_closed_threshold=[0.0, 0.025],
                 on_z_threshold=[0.047, 0.053], xyz_tgt_threshold=[0.0,0.05]):
    preds, n_hots = [], []
    for o,g in zip(obs,goal):
        p,nh = obs_to_preds_single(o,g,n_objects, grasp_xy_threshold, grasp_z_threshold,
                 grip_open_threshold, grip_closed_threshold,
                 on_z_threshold, xyz_tgt_threshold)
        preds.append(p)
        n_hots.append(nh)

    return preds, n_hots


def obs_to_preds_single(obs, goal, n_objects,
                 grasp_xy_threshold=[0.0, 0.025], grasp_z_threshold=[-0.015, 0.02],
                 grip_open_threshold=[0.038, 1.0], grip_closed_threshold=[0.0, 0.025],
                 on_z_threshold=[0.047, 0.053], xyz_tgt_threshold=[0.0,0.05]):


    preds = {}
    gripper_pos = obs[0:3]
    gripper_state = np.sum(obs[3 + 6 * n_objects: 3 + 6 * n_objects + 1])
    gripper_in_goal = (len(goal) / 3) > n_objects

    def get_o_pos(obs, o_idx):
        start_idx = (o_idx + 1) * 3
        end_idx = start_idx + 3
        o_pos = obs[start_idx:end_idx]
        return o_pos

    def get_o_goal_pos(goal, o_idx):
        start_idx = (o_idx + 1) * 3
        if gripper_in_goal is False:
            start_idx -= 3
        end_idx = start_idx + 3
        g_pos = goal[start_idx:end_idx]
        return g_pos

    # Determine whether gripper has reached an object's location
    for o in range(n_objects):
        pred_name = 'gripper_at_o{}'.format(o)
        o_pos = get_o_pos(obs, o)
        xyd = np.linalg.norm(gripper_pos[:2] - o_pos[:2], axis=-1)
        xyd_ok = int(xyd > grasp_xy_threshold[0] and xyd < grasp_xy_threshold[1])
        zd = gripper_pos[2] - o_pos[2]
        zd_ok = int(zd > grasp_z_threshold[0] and zd < grasp_z_threshold[1])
        reached = int(xyd_ok and zd_ok)
        preds[pred_name] = reached

    # Determine whether an object is on top of another object
    for o1 in range(n_objects):
        o1_pos = get_o_pos(obs, o1)
        for o2 in range(n_objects):
            if o1 == o2:
                continue
            pred_name = 'o{}_on_o{}'.format(o1,o2)
            o2_pos = get_o_pos(obs, o2)
            xyd = np.linalg.norm(o1_pos[:2] - o2_pos[:2], axis=-1)
            xyd_ok = int(xyd > grasp_xy_threshold[0] and xyd < grasp_xy_threshold[1])
            zd = o2_pos[2] - o1_pos[2]
            zd_ok = int(zd > on_z_threshold[0] and zd < on_z_threshold[1])
            on = int(xyd_ok and zd_ok)
            preds[pred_name] = on

    # Determine open and closed state of the gripper
    preds['gripper_open'] = int(gripper_state > grip_open_threshold[0] and gripper_state < grip_open_threshold[1])
    preds['gripper_closed'] = int(gripper_state > grip_closed_threshold[0] and gripper_state < grip_closed_threshold[1])

    for o in range(n_objects):
        pred_name = 'grasped_o{}'.format(o)
        preds[pred_name] = int(preds['gripper_at_o{}'.format(o)] == 1 and preds['gripper_closed'] == 1)
        pred_name = 'o{}_at_target'.format(o)
        o_pos = get_o_pos(obs, o)
        g_pos = get_o_goal_pos(goal, o)
        xyzd = np.linalg.norm(g_pos - o_pos, axis=-1)
        preds[pred_name] = int(xyzd >= xyz_tgt_threshold[0] and xyzd <= xyz_tgt_threshold[1])

    if gripper_in_goal:
        xyzd = np.linalg.norm(gripper_pos - goal[:3], axis=-1)
        preds['gripper_at_target'] = int(xyzd >= xyz_tgt_threshold[0] and xyzd <= xyz_tgt_threshold[1])
    else:
        preds['gripper_at_target'] = 1

    one_hot = [preds[k] for k in sorted(preds.keys())]
    return preds, one_hot


def gen_pddl_domain_problem(preds, tower_height, gripper_has_target=True):

    head = "(define (domain tower) (:requirements :strips) \n{})\n"

    predicates = "(:predicates \n"
    for p in sorted(preds.keys()):
        predicates += "\t{}\n".format(p)
    predicates += ")\n"

    n_objects = len([p for p in preds if p.find("gripper_at_o") != -1])

    actions = []
    not_grasped_str = ''
    for o in range(n_objects):
        not_grasped_str += '(not (grasped_o{}))'.format(o)
    open_gripper_act = "(:action open_gripper \n\t:parameters () \n\t:precondition () \n\t:effect (and (gripper_open) {} )\n)\n\n".format(not_grasped_str)
    actions.append(open_gripper_act)
    grasp_act_template = "(:action grasp__o{} \n\t:parameters () \n\t:precondition (and (gripper_at_o{}) {}) \n\t:effect (and (grasped_o{}) (not (gripper_open)))\n)\n\n"
    move_gripper_to_o_act_template = "(:action move_gripper_to__o{} \n\t:parameters () \n\t:precondition (gripper_open) \n\t:effect (and (gripper_at_o{}) (not (gripper_at_target)) {})\n)\n\n"
    move_o1_on_o2_act_template = "(:action move__o{}_on__o{} \n\t:parameters () \n\t:precondition (and (grasped_o{}) {}) \n\t:effect (o{}_on_o{})\n)\n\n"
    move_o_to_target_template = "(:action move__o{}_to_target \n\t:parameters () \n\t:precondition (and (grasped_o{}) {}) \n\t:effect (o{}_at_target)\n)\n\n"
    move_o_on_table_template = "(:action move__o{}_on_table \n\t:parameters () \n\t:precondition (and (grasped_o{})) \n\t:effect (and (not (o{}_at_target)) {} )\n)\n\n"

    for o in range(n_objects):
        # Grasp object action
        not_o2_on_o_str = ''
        for o2 in range(n_objects):
            if o == o2:
                continue
            not_o2_on_o_str += ' (not (o{}_on_o{}))'.format(o2, o)
        grasp_act = grasp_act_template.format(o, o, not_o2_on_o_str, o)
        actions.append(grasp_act)

        # Move gripper to object action
        not_elsewhere_str = ''
        for o_other in range(n_objects):
            if o_other == o:
                continue
            not_elsewhere_str += '(not (gripper_at_o{}))'.format(o_other)

        move_gripper_to_o_act = move_gripper_to_o_act_template.format(o, o, not_elsewhere_str)
        actions.append(move_gripper_to_o_act)

        # Move o to target action. This is to place the first object on the ground on which other objects will be stacked.
        not_o2_on_o_str = ''
        for o2 in range(n_objects):
            if o == o2:
                continue
            not_o2_on_o_str += ' (not (o{}_on_o{}))'.format(o2, o)
        move_o_to_target_act = move_o_to_target_template.format(o, o, not_o2_on_o_str, o)
        actions.append(move_o_to_target_act)

        # Move o1 on o2 action. This is to stack objects on other objects.
        for o2 in range(n_objects):
            if o+1 != o2:
                continue # To restrict the action space, only allow to put an object with number o onto o+1 (e.g. object 0 on object 1, but not object 0 on object 2.)
            not_o3_on_o2_str = ''
            for o3 in range(n_objects):
                if o3 == o2:
                    continue
                not_o3_on_o2_str += ' (not (o{}_on_o{}))'.format(o3, o2)
            move_o1_on_o2_act = move_o1_on_o2_act_template.format(o, o2, o, not_o3_on_o2_str, o, o2)
            actions.append(move_o1_on_o2_act)

        # Move_o_on_table_action
        not_o_on_o2_str = ''
        for o2 in range(n_objects):
            if o == o2:
                continue
            not_o_on_o2_str += ' (not (o{}_on_o{}))'.format(o, o2)
        move_on_table_act = move_o_on_table_template.format(o, o, o, not_o_on_o2_str)
        actions.append(move_on_table_act)


        # Move o to target action
        # move_o_to_tgt_act = move_o_to_target_template.format(o, o, o)
        # actions.append(move_o_to_tgt_act)

    not_elsewhere_str = ''
    for o in range(n_objects):
        not_elsewhere_str += '(not (gripper_at_o{}))'.format(o)
    move_gripper_to_target = "(:action move_gripper_to_target \n\t:parameters () \n\t:precondition (gripper_open) \n\t:effect (and (gripper_at_target) {})\n)\n\n".format(not_elsewhere_str)

    actions.append(move_gripper_to_target)
    body = predicates + "".join(actions)
    domain = head.format(body)


    # Define problem
    problem = "(define (problem pb1) (:domain tower)\n {}\n)\n"

    # Define initial state
    init_str = "(:init\n"
    for pred in preds:
        pred_val = pred if preds[pred] == 1 else "not ({})".format(pred)
        init_str+= "\t ({})\n".format(pred_val)
    init_str += ")\n"

    # Define goal
    tgt_goal_z_idx = 5
    if gripper_has_target == False:
        tgt_goal_z_idx -= 3
    # tower_height = round((goal[tgt_goal_z_idx] - table_height + (block_height / 2)) / block_height)
    # tower_height = int(tower_height)
    goal_str = "(:goal (and \n"
    if gripper_has_target:
        pred_val = "o{}_at_target".format(tower_height - 1)
        goal_str += "\t ({})\n".format(pred_val)
        for o in range(tower_height - 1):
            pred_val = "o{}_on_o{}".format(o, o + 1)
            goal_str += "\t ({})\n".format(pred_val)

        pred_val = "gripper_at_target"
        goal_str += "\t ({})\n".format(pred_val)
    else:
        pred_val = "o{}_at_target".format(tower_height - 1)
        goal_str += "\t ({})\n".format(pred_val)
        for o in range(tower_height - 1):
            pred_val = "o{}_on_o{}".format(o, o + 1)
            goal_str += "\t ({})\n".format(pred_val)
    goal_str += '))\n\n'

    problem = problem.format(init_str + goal_str)

    return domain, problem

def gen_plans(preds, gripper_has_target, tower_height):
    plans = []
    for p in preds:
        plan = gen_plan_single(p, gripper_has_target, tower_height)
        plans.append(plan)
    return plans

def gen_plan_single(preds, gripper_has_target, tower_height):

    # dom_file = None
    # prob_file = None

    hl_obs = [preds[k] for k in sorted(preds.keys())]
    prob_key = str(hl_obs) + "_" + str(tower_height)

    # dom_file = 'pddl/build_tower/{}_domain.pddl'.format(prob_key)
    # prob_file = 'pddl/build_tower/{}_problem.pddl'.format(prob_key)

    domain, problem = gen_pddl_domain_problem(preds, tower_height, gripper_has_target=gripper_has_target)

    planner = Propositional_Planner()
    plan_start = time.time()
    plan = planner.solve(domain, problem)
    # logger.info("plan generation took {:.2f} sec.".format(time.time() - plan_start))
    plan_acts = []
    if plan:
        # print('plan:')
        for i, act in enumerate(plan):
            print('Act {}: {}'.format(i, act))
            plan_acts.append(act.name)
    else:
        print('No plan was found')


    # Generate vector encodings of plan
    act_names = []
    dom_lines = domain.split("\n")
    for dl in dom_lines:
        if dl.find("(:action ") != -1:
            act = dl.split("(:action ")[1].strip()
            act_names.append(act)
    act_names = sorted(act_names)

    # Generate one-hot encoding of plan
    one_hot_arr_plan = []
    for plan_act in plan_acts:
        one_hot_arr = np.zeros(len(act_names))
        for idx, act in enumerate(act_names):
            if act == plan_act:
                one_hot_arr[idx] = 1
        one_hot_arr_plan.append(one_hot_arr)
    one_hot_arr_plan.append(np.zeros(len(act_names)))

    # Generate n-hot encoding of plan
    n_hot_arr_plan = []
    arg1_names = []
    act_templates = []
    for an in act_names:
        act, arg1 = get_act_name_arg1(an)
        if arg1 != '':
            arg1_names.append(arg1)
        act_templates.append(act)

    act_templates = sorted(list(set(act_templates)))
    arg1_names = sorted(list(set(arg1_names)))

    for plan_act in plan_acts:
        n_hot_arr = np.zeros(len(act_templates) + len(arg1_names))
        act, arg1 = get_act_name_arg1(plan_act)
        # set action template index to 1
        idx = act_templates.index(act)
        n_hot_arr[idx] = 1
        # set arg1 index to 1
        if arg1 != '':
            idx = arg1_names.index(arg1) + len(act_templates)
        else:
            idx = 0
        n_hot_arr[idx] = 1
        n_hot_arr_plan.append(n_hot_arr)
    n_hot_arr_plan.append(np.zeros(len(act_templates) + len(arg1_names)))

    return plan_acts, one_hot_arr_plan, n_hot_arr_plan


# Returns the name of an action and its first argument.
def get_act_name_arg1(param_act_name):
    arg1_start_pos = param_act_name.find("__")
    if arg1_start_pos == -1:
        return param_act_name, ''
    arg1_end_pos = param_act_name.find("_", arg1_start_pos+2)
    if arg1_end_pos == -1:
        arg1_end_pos = len(param_act_name)
    arg_name = param_act_name[arg1_start_pos+2:arg1_end_pos]
    act_name = param_act_name.replace("__"+arg_name, '', 1)

    return act_name, arg_name



