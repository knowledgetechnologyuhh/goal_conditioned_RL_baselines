import numpy as np
from baselines.her_pddl.pddl.propositional_planner import Propositional_Planner
from gym.envs.robotics import rotations
import time

import copy


class RakeObjectThresholds: #TODO: fix this
    grip_open_threshold = [0.038, 1.0]
    grip_closed_threshold = [0.0, 0.025]
    distance_threshold = 0.02
    grasp_z_offset = 0.005
    rake_handle_x_offset = 2*(0.125-0.01)
    at_x_offset = 0.025
    at_y_offset = 0.03
    on_z_offset = 0.05


def get_o_pos(obs, o_idx):
    start_idx = (o_idx + 1) * 3
    end_idx = start_idx + 3
    o_pos = obs[start_idx:end_idx]
    return o_pos


def get_o_rot(obs, o_idx, n_objects=2):
    start_idx = (o_idx + 2 + 2*n_objects) * 3 - 1 # gripper_state has 2 in size
    end_idx = start_idx + 3
    o_rot = obs[start_idx:end_idx]
    return o_rot


def compute_handle_pos(tip_pos, tip_rot):
    rot_mat = rotations.euler2mat(tip_rot)
    tran_mat = np.zeros((4, 4))
    tran_mat[3, 3] = 1.
    tran_mat[:3, 3] = tip_pos
    tran_mat[:3, :3] = rot_mat
    handle = np.zeros((4,))
    # handle[:3] = subgoal[:3]
    handle[0] = -RakeObjectThresholds.rake_handle_x_offset
    # handle[3] = 1.
    handle_new = tran_mat * handle
    return handle_new[:3, 0]


def obs_to_preds(obs, goal, n_objects):
    preds, n_hots = [], []
    for o, g in zip(obs, goal):
        p, oh = obs_to_preds_single(o, g, n_objects)
        preds.append(p)
        n_hots.append(oh)
    return preds, np.array(n_hots)


def obs_to_preds_single(obs, goal, n_objects):  # TODO: check
    ROT = RakeObjectThresholds
    preds = {}
    gripper_pos = obs[0:3]
    gripper_state = np.sum(obs[3 + 6 * n_objects: 3 + 6 * n_objects + 1])
    gripper_in_goal = (len(goal) / 3) > n_objects

    # def get_o_pos(obs, o_idx):
    #     start_idx = (o_idx + 1) * 3
    #     end_idx = start_idx + 3
    #     o_pos = obs[start_idx:end_idx]
    #     return o_pos.copy()

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
        gripper_tgt_pos = o_pos.copy()
        if o == 0: # if the hook, assuming that the hook position returned from the environment is at the tip
            # gripper_tgt_pos[0] -= ROT.rake_handle_x_offset  # the hook is 0.2 m long in x-axis
            o_rot = get_o_rot(obs, o, n_objects)
            handle_offset = compute_handle_pos(o_pos, o_rot)
            gripper_tgt_pos += handle_offset
        gripper_tgt_pos[2] += ROT.grasp_z_offset
        distance = np.linalg.norm(gripper_pos - gripper_tgt_pos)
        preds[pred_name] = distance < ROT.distance_threshold

    # Determine whether an object is on top of another object --> old
    # Determine whether the rake/hook (object0) near the cube (object1) TODO: fix
    # for o1 in range(n_objects):
    #     o1_pos = get_o_pos(obs, o1)
    #     for o2 in range(n_objects):
    #         if o1 == o2:
    #             continue
    #         pred_name = 'o{}_on_o{}'.format(o1, o2)
    #         o2_pos = get_o_pos(obs, o2)
    #         o1_tgt_pos = o2_pos + [0, 0, ROT.on_z_offset]
    #         distance = np.linalg.norm(o1_tgt_pos - o1_pos)
    #         preds[pred_name] = distance < ROT.distance_threshold

    # Determine whether the rake/hook (object0) at the cube (object1)
    # --> environment is defined in such a way that the rake end-effector is always front of the cube wrt the robot
    for o1 in range(1):
        o1_pos = get_o_pos(obs, o1)
        # o1_pos[0] += ROT.rake_handle_x_offset   # use the tip of the hook to find the distance to the cube
        for o2 in range(1, n_objects):
            if o1 == o2:
                continue
            pred_name = 'o{}_at_o{}'.format(o1, o2)
            # TODO: fix this if the hook position returned from the environment is at the center of mass
            o2_pos = get_o_pos(obs, o2)
            # o1_tgt_pos = o2_pos + [0, 0, ROT.on_z_offset]
            distance = np.linalg.norm(o2_pos - o1_pos)
            preds[pred_name] = distance < 4.*ROT.distance_threshold

    for o in range(n_objects):
        pred_name = 'o{}_at_target'.format(o)
        o_pos = get_o_pos(obs, o)
        g_pos = get_o_goal_pos(goal, o)
        distance = np.linalg.norm(g_pos - o_pos, axis=-1)
        threshold = ROT.distance_threshold if o == 0 else 1.5*ROT.distance_threshold
        preds[pred_name] = distance < threshold

    if gripper_in_goal:
        distance = np.linalg.norm(gripper_pos - goal[:3], axis=-1)
        preds['gripper_at_target'] = distance < ROT.distance_threshold
    else:
        preds['gripper_at_target'] = 1

    preds = {p: int(v) for p,v in preds.items()}
    one_hot = np.array([preds[k] for k in sorted(preds.keys())])

    # goal_preds = []
    # if gripper_in_goal:
    #     start_idx = 3
    # else:
    #     start_idx = 0

    # goal_o_order = {}
    # goal_pos = {}
    # for o_idx in range(n_objects):
    #     o_z = get_o_goal_pos(goal, o_idx)[2]
    #     z_offset = 0.5
    #     z_delta = 0.05
    #     o_z -= (z_offset + z_delta/2)
    #     n_order = o_z / z_delta
    #     n_order = int(round(n_order))
    #     # n_order = int(n_order)
    #     goal_o_order[o_idx] = n_order
    #     goal_pos[n_order] = o_idx
    # print('goal_pos: {}'.format(goal_pos))
    goal_preds = []
    # for pos, o_idx in goal_pos.items():
    #     if pos == 0:
    #         goal_pred_str = "o{}_at_target".format(o_idx)
    #     else:
    #         goal_pred_str = "o{}_at_o{}".format(o_idx, goal_pos[pos - 1])
    #     goal_preds.append(goal_pred_str)
    for o in range(n_objects):
        goal_pred_str = "o{}_at_target".format(o)
        goal_preds.append(goal_pred_str)
    if gripper_in_goal:
        goal_preds.append('gripper_at_target')

    return preds, one_hot, goal_preds


def gen_actions(n_objects): # TODO: check
    actions = []
    not_grasped_str = ''
    for o in range(n_objects):
        not_grasped_str += '(not (grasped_o{}))'.format(o)
    move_gripper_to_o_act_template = \
        "(:action move_gripper_to__o{} \n\t" \
        ":parameters () \n\t" \
        ":precondition () \n\t" \
        ":effect (and (gripper_at_o{}) {} {} (not (gripper_at_target)) )\n)\n\n"
    move_o_to_target_by_o_act_template = \
        "(:action move__o{}_to_target_by__o{} \n\t" \
        ":parameters () \n\t" \
        ":precondition (and (o{}_at_o{}) ) \n\t" \
        ":effect (and (o{}_at_target) (o{}_at_o{}) )\n)\n\n"
      # ":effect (and (o{}_at_target) (o{}_at_o{}) (gripper_at_o{}))\n)\n\n"
      # ":precondition (and (o{}_at_o{}) (gripper_at_o{})) \n\t"
    move_o_to_target_act_template = \
        "(:action move__o{}_to_target \n\t" \
        ":parameters () \n\t" \
        ":precondition (and (gripper_at_o{}) ) \n\t" \
        ":effect (and (o{}_at_target) (gripper_at_o{}))\n)\n\n"
    move_o1_on_o2_act_template = \
        "(:action move__o{}_on__o{}  \n\t" \
        ":parameters () \n\t" \
        ":precondition (and (gripper_at_o{}) ) \n\t" \
        ":effect (and (o{}_on_o{})  {} )\n)\n\n"
    move_o1_at_o2_act_template = \
        "(:action move__o{}_at__o{}  \n\t" \
        ":parameters () \n\t" \
        ":precondition (and (gripper_at_o{}) ) \n\t" \
        ":effect (and (o{}_at_o{}) )\n)\n\n"
      # ":effect (and (o{}_at_o{}) (gripper_at_o{}))\n)\n\n"

    not_o2_at_o_str = ''
    not_elsewhere_str = ''
    for o in range(n_objects):
        # Grasp object action
        # not_o2_at_o_str = ''
        for o2 in range(n_objects):
            if o == o2:
                continue
            not_o2_at_o_str += ' (not (o{}_at_o{}))'.format(o2, o)  # TODO: check
        # not_elsewhere_str = ''
        for o_other in range(n_objects):
            if o_other == o:
                continue
            if o_other != 0:
                not_elsewhere_str += '(not (gripper_at_o{}))'.format(o_other)

        # # Move gripper to object action
        # move_gripper_to_o_act = move_gripper_to_o_act_template.format(o, o, not_elsewhere_str, not_o2_at_o_str)
        # actions.append(move_gripper_to_o_act)
        #
        # # Move o to target action.
        # # This is to place the first object on the ground on which other objects will be stacked.
        # move_o_to_target_act = move_o_to_target_template.format(o, o, o)
        # actions.append(move_o_to_target_act)
        #
        # # Move o1 on o2 action. This is to stack objects on other objects.
        # for o2 in range(n_objects):
        #     # if o-1 != o2:
        #     #     continue # To restrict the action space, only allow to put an object with number o onto o-1
        #     #     (e.g. object 1 on object 0, but not object 0 on object 2.)
        #     not_o3_on_o2_str = ''
        #     for o3 in range(n_objects):
        #         if o3 == o2:
        #             continue
        #         if o3 == o:
        #             continue
        #         not_o3_on_o2_str += ' (not (o{}_on_o{}))'.format(o3, o2)
        #     move_o1_on_o2_act = move_o1_on_o2_act_template.format(o, o2, o, o, o2, not_o3_on_o2_str)
        #     actions.append(move_o1_on_o2_act)
    # Move gripper to hook (object0) action --> TODO: check where can grasp the hook
    move_gripper_to_hook_act = move_gripper_to_o_act_template.format(0, 0, not_elsewhere_str, not_o2_at_o_str)
    actions.append(move_gripper_to_hook_act)

    if n_objects>=2:
        # Move the hook to the cube (object1) action.
        # This is to place the first object on the ground on which other objects will be stacked.
        # move_hook_to_o_act = move_o1_at_o2_act_template.format(0, 1, 0, 0, 1, 0)
        move_hook_to_o_act = move_o1_at_o2_act_template.format(0, 1, 0, 0, 1)
        actions.append(move_hook_to_o_act)

        # Move o to target action.
        # This is to place the cube on the ground at the target position
        # move_o_to_target_act = move_o_to_target_by_o_act_template.format(1, 0, 0, 1, 0, 1, 0, 1, 0)
        # move_o_to_target_act = move_o_to_target_by_o_act_template.format(1, 0, 0, 1, 1, 0, 1, 0)
        move_o_to_target_act = move_o_to_target_by_o_act_template.format(1, 0, 0, 1, 1, 0, 1)
        actions.append(move_o_to_target_act)

        move_hook_to_target_act = move_o_to_target_act_template.format(0, 0, 0, 0)
        actions.append(move_hook_to_target_act)
    else:
        move_o_to_target_act = move_o_to_target_act_template.format(0, 0, 0, 0)
        actions.append(move_o_to_target_act)

    # This is to place the hook to its target position  --> TODO: check if necessary
    # move_hook_to_target_act = move_o_to_target_template.format(0, 0, 1, 0)
    # actions.append(move_hook_to_target_act)

    not_elsewhere_str = ''
    for o in range(n_objects):
        not_elsewhere_str += '(not (gripper_at_o{}))'.format(o)
    move_gripper_to_target = \
        "(:action move_gripper_to_target \n\t" \
        ":parameters () \n\t" \
        ":precondition (and {}) \n\t" \
        ":effect (and (gripper_at_target) {})\n)\n\n".format(not_grasped_str, not_elsewhere_str)

    actions.append(move_gripper_to_target)
    return actions


def gen_pddl_domain_problem(preds, goal_preds, gripper_has_target=True):

    head = "(define (domain hook) (:requirements :strips) \n{})\n"

    predicates = "(:predicates \n"
    for p in sorted(preds.keys()):
        predicates += "\t{}\n".format(p)
    predicates += ")\n"

    n_objects = len([p for p in preds if p.find("gripper_at_o") != -1])

    actions = gen_actions(n_objects)
    body = predicates + "".join(actions)
    domain = head.format(body)

    # Define problem
    problem = "(define (problem pb1) (:domain hook)\n {}\n)\n"

    # Define initial state
    init_str = "(:init\n"
    for pred in preds:
        pred_val = pred if preds[pred] == 1 else "not ({})".format(pred)
        init_str+= "\t ({})\n".format(pred_val)
    init_str += ")\n"

    # Define goal
    goal_str = "(:goal (and \n"
    # o_order = []
    for g in goal_preds:
        goal_str += '\t ({})\n'.format(g)
    goal_str += '))\n\n'

    problem = problem.format(init_str + goal_str)

    return domain, problem


# Seem not to be used # TODO: check
def gen_plans(preds, gripper_has_target, tower_height):
    plans = []
    for p in preds:
        plan = gen_plan_single(p, gripper_has_target, tower_height)
        plans.append(plan)
    return plans


# Seem not to be used TODO: check
def plans2subgoals(plans, obs, goals, n_objects, actions_to_skip=[]):   # TODO: fix this
    subgoals = np.zeros(goals.shape)
    for i, (p, o, g) in enumerate(zip(plans, obs, goals)):
        subgoal = plan2subgoal(p, o, g, n_objects, actions_to_skip=actions_to_skip)
        subgoals[i] = subgoal
    return subgoals


def action2subgoal(action, obs, goal, n_objects):
    ROT = RakeObjectThresholds

    # def get_o_pos(obs, o_idx):
    #     start_idx = (o_idx + 1) * 3
    #     end_idx = start_idx + 3
    #     o_pos = obs[start_idx:end_idx]
    #     return o_pos
    #
    # def get_o_rot(obs, o_idx):
    #     start_idx = (o_idx + 6) * 3 - 1 # gripper_state has 2 in size
    #     end_idx = start_idx + 3
    #     o_rot = obs[start_idx:end_idx]
    #     return o_rot
    #
    # def compute_handle_pos(tip_pos, tip_rot):
    #     rot_mat = rotations.euler2mat(tip_rot)
    #     tran_mat = np.zeros((4, 4))
    #     tran_mat[3, 3] = 1.
    #     tran_mat[:3, 3] = tip_pos
    #     tran_mat[:3, :3] = rot_mat
    #     handle = np.zeros((4,))
    #     # handle[:3] = subgoal[:3]
    #     handle[0] = -ROT.rake_handle_x_offset
    #     # handle[3] = 1.
    #     handle_new = tran_mat * handle
    #     return handle_new[:3, 0]

    final_goal = copy.deepcopy(goal)
    # subgoal = copy.deepcopy(goal)
    no_change_subgoal = copy.deepcopy(goal)
    # By default, all objects stays where they are:
    for o_idx in range(n_objects):
        o_pos = get_o_pos(obs, o_idx).copy()
        start_idx = (o_idx + 1) * 3
        end_idx = start_idx + 3
        no_change_subgoal[start_idx:end_idx] = o_pos
    subgoal = no_change_subgoal.copy()
    for o_idx in range(n_objects):
        o_pos = get_o_pos(obs, o_idx)
        o_rot = get_o_rot(obs, o_idx, n_objects)
        # print('o_rot {}'.format(o_rot))
        o_rot_deg = o_rot * 180. / np.pi
        if action == 'move_gripper_to__o{}'.format(o_idx):
            # First three elements of goal represent target gripper pos.
            subgoal[:3] = o_pos.copy()  # Gripper should be above the handle of the hook
            if o_idx == 0:  # if the hook, assuming that the hook position returned from the environment is at the tip
                # subgoal[0] -= ROT.rake_handle_x_offset  # the hook is 0.38 m long in x-axis
                # theta = o_rot[2]
                # subgoal[0] -= ROT.rake_handle_x_offset * np.cos(theta)  # the hook is 0.38 m long in x-axis
                # subgoal[1] += ROT.rake_handle_x_offset * np.sin(theta)  # the hook is 0.38 m long in x-axis
                # rot_mat = rotations.euler2mat(o_rot)
                # tran_mat = np.zeros((4, 4))
                # tran_mat[3, 3] = 1.
                # tran_mat[:3, 3] = o_pos
                # tran_mat[:3, :3] = rot_mat
                # handle = np.zeros((4,))
                # # handle[:3] = subgoal[:3]
                # handle[0] = -ROT.rake_handle_x_offset
                # # handle[3] = 1.
                # handle_new = tran_mat*handle
                # subgoal[:3] += handle_new[:3, 0]
                handle_offset = compute_handle_pos(o_pos, o_rot)
                subgoal[:3] += handle_offset
            subgoal[2] += np.mean(ROT.grasp_z_offset)
        if action == 'move__o{}_to_target_by__o{}'.format(o_idx, o_idx-1) and o_idx == 1:
            hook_pos = get_o_pos(obs, 0)
            # The cube is at its target
            start_idx = (o_idx + 1) * 3
            end_idx = start_idx + 3
            o_goal = goal[start_idx:end_idx]
            # Object should be at object goal
            subgoal[start_idx:end_idx] = o_goal
            # The hook is near the cube
            o2_pos = o_goal.copy()
            o2_pos[0] += ROT.at_x_offset
            if hook_pos[1] >= o_goal[1]:  # the hook is on the left side of the cube
                o2_pos[1] += ROT.at_y_offset
            else:
                o2_pos[1] -= ROT.at_y_offset
            subgoal[o_idx*3:o_idx*3+3] = o2_pos
            # Gripper should be at the handle of the hook
            subgoal[:3] = o2_pos.copy()
            # o2_rot = get_o_rot(obs, 0)
            # subgoal[:3] += compute_handle_pos(o2_pos, o2_rot)
            subgoal[2] += np.mean(ROT.grasp_z_offset)
            subgoal[0] -= ROT.rake_handle_x_offset  # the hook is 0.38 m long in x-axis

        if action == 'move__o{}_to_target'.format(o_idx) and o_idx == 0:
            start_idx = (o_idx + 1) * 3
            end_idx = start_idx + 3
            o_goal = goal[start_idx:end_idx]
            # Object (hook) should be at object goal
            subgoal[start_idx:end_idx] = o_goal

            # Gripper should be at the handle of the hook
            subgoal[:3] = o_goal.copy()
            subgoal[2] += np.mean(ROT.grasp_z_offset)
            subgoal[0] -= ROT.rake_handle_x_offset  # the hook is 0.38 m long in x-axis

        for o2_idx in range(n_objects):
            o2_pos = get_o_pos(obs, o2_idx)
            if action == 'move__o{}_at__o{}'.format(o_idx, o2_idx):
                # First three elements of goal represent target gripper pos.
                # subgoal[:3] = o2_pos.copy()  # Gripper should be above (at) object
                # subgoal[2] += np.mean(BTT.grasp_z_offset)
                # Hook end-effector should be near the cube (o2)
                o_goal = o2_pos.copy()
                o_goal[0] += ROT.at_x_offset
                if o_pos[1] >= o2_pos[1]: # the hook is on the left side of the cube
                    o_goal[1] += ROT.at_y_offset
                else:
                    o_goal[1] -= ROT.at_y_offset
                # o_goal[2] += (ROT.on_z_offset * 1.0)
                start_idx = (o_idx + 1) * 3
                end_idx = start_idx + 3
                subgoal[start_idx:end_idx] = o_goal # hook position
                subgoal[:3] = o_goal                # gripper position
                if o_idx == 0:  # if the hook
                    subgoal[0] -= ROT.rake_handle_x_offset  # the hook is 0.38 m long in x-axis
                subgoal[2] += np.mean(ROT.grasp_z_offset)
    if action == 'move_gripper_to_target':
        subgoal[3:] = final_goal[3:]  # Gripper should be at gripper goal

    if str(subgoal) == str(no_change_subgoal):
        print("Warning, looks like action {} is invalid. Setting subgoal to final goal".format(action))
        subgoal = goal.copy()
    return subgoal


# Seem to be used only in plans2subgoal TODO: check
def plan2subgoal(plan, obs, goal, n_objects, actions_to_skip = []): # TODO: check
    for action in plan[0]:
        if action in actions_to_skip:
            continue
        subgoal = action2subgoal(action, obs, goal, n_objects)
        break  # Stop after first useful action has been found.
    return subgoal


def gen_plan_single(preds, gripper_has_target, goal_preds, ignore_actions=[]):

    # hl_obs = [preds[k] for k in sorted(preds.keys())]

    domain, problem = gen_pddl_domain_problem(preds, goal_preds, gripper_has_target=gripper_has_target)
    # print('domain {}'.format(domain))
    # print('problem {}'.format(problem))

    planner = Propositional_Planner()
    plan_start = time.time()
    plan = planner.solve(domain, problem)
    duration = time.time() - plan_start
    if duration > 10.0:
        print("Plan generation took {:.2f} sec.".format(time.time() - plan_start))

    plan_acts = []
    goal_achieved = False
    if plan is None:
        print('No plan was found')
        with open('no_plan_dom.pddl', 'w') as f:
            f.write(domain)
        with open('no_plan_prob.pddl', 'w') as f:
            f.write(problem)
        goal_achieved = False
    elif plan == []:
        goal_achieved = True
        # print("Goal already achieved")
    else:
        # print('plan:')
        for i, act in enumerate(plan):
            # print('Act {}: {}'.format(i, act))
            if act.name not in ignore_actions:
                plan_acts.append(act.name)

    # Generate vector encodings of plan
    act_names = []
    dom_lines = domain.split("\n")
    for dl in dom_lines:
        if dl.find("(:action ") != -1:
            act = dl.split("(:action ")[1].strip()
            if act not in ignore_actions:
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

    return plan_acts, one_hot_arr_plan, n_hot_arr_plan, goal_achieved


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



