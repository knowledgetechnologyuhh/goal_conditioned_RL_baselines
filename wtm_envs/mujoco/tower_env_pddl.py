import numpy as np
from baselines.her_pddl.pddl.propositional_planner import Propositional_Planner
# from wtm_envs.mujoco.pddl_env import PDDLExtension
import time
from wtm_envs.mujoco.pddl_env import PDDLEnv

class PDDLTowerEnv(PDDLEnv):
    grip_open_threshold = [0.038, 1.0]
    grip_closed_threshold = [0.0, 0.025]
    distance_threshold = 0.025
    grasp_z_offset = 0.02
    on_z_offset = 0.05
    table_height = 0.525

    def __init__(self, n_objects, gripper_has_target):
        self.n_objects = n_objects
        self.gripper_has_target = gripper_has_target
        PDDLEnv.__init__(self)

    def get_goal(self):
        goal_preds = self.goal2preds(g, self.n_objects)
        return goal_preds

    def get_plan(self, return_states=False):
        obs = self._get_obs()
        if self.final_goal == []:
            g = self.goal
        else:
            g = self.final_goal
        obs_preds, obs_n_hots = self.obs2preds_single(obs['observation'], g)
        padded_goal_obs = self._goal2obs(g)
        goal_preds, goal_n_hots = self.obs2preds_single(padded_goal_obs, g)


        cache_key = str(obs_n_hots) + str(goal_n_hots)
        if cache_key in self.plan_cache.keys():
            plan, world_states = self.plan_cache[cache_key]
        else:
            plan, world_states = self.gen_plan_single(obs_preds, self.gripper_has_target, goal_preds)

            self.plan_cache[cache_key] = (plan, world_states)
            if len(self.plan_cache) % 50 == 0:
                print("Number of cached plans: {}".format(len(self.plan_cache)))
        if return_states:
            return plan, world_states
        else:
            return plan

    def preds2subgoal(self, preds):
        obs = self._get_obs()['observation']
        subgoal_obs = self.preds2subgoal_obs(preds, obs, self.final_goal)
        subg = self._obs2goal(subgoal_obs)
        return subg


    def _gen_pred_functs(self):
        self.pred2subg_functs = {}
        self.obs2pred_functs = {}

        def make_gripper_at_o_functs(o_idx):

            def _pred2subg_function(obs, goal):
                o_pos = self.get_o_pos(obs, o_idx)
                gripper_tgt_pos = o_pos.copy()
                gripper_tgt_pos[2] += self.grasp_z_offset
                subg = [0] + list(gripper_tgt_pos)
                return subg

            def _obs2pred_function(obs, goal):
                gripper_tgt_pos = _pred2subg_function(obs, goal)[1:]
                gripper_pos = obs[0:3]
                distance = np.linalg.norm(gripper_pos - gripper_tgt_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        for o in range(self.n_objects):
            pred_name = 'gripper_at_o{}'.format(o)
            self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] = make_gripper_at_o_functs(o)

        def make_o_on_o_functs(o1_idx, o2_idx):
            def _pred2subg_function(obs, goal):
                o2_pos = self.get_o_pos(obs, o2_idx)
                o1_tgt_pos = o2_pos + [0, 0, self.on_z_offset]
                subg = [o1_idx + 1] + list(o1_tgt_pos)
                return subg

            def _obs2pred_function(obs, goal):
                tgt_pos = _pred2subg_function(obs, goal)[1:]
                o_pos = self.get_o_pos(obs, o1_idx)
                distance = np.linalg.norm(o_pos - tgt_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        for o1 in range(self.n_objects):
            for o2 in range(self.n_objects):
                if o1 == o2:
                    continue
                pred_name = 'o{}_on_o{}'.format(o1, o2)
                self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] = make_o_on_o_functs(o1, o2)

        def make_gripper_tgt_funct():
            def _pred2subg_function(obs, goal):
                gripper_tgt_pos = goal[0:3]
                subg = [0] + list(gripper_tgt_pos)
                return subg

            def _obs2pred_function(obs, goal):
                tgt_pos = _pred2subg_function(obs, goal)[1:]
                gripper_pos = obs[0:3]
                distance = np.linalg.norm(gripper_pos - tgt_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        if self.gripper_has_target:
            pred_name = 'gripper_at_target'
            self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] = make_gripper_tgt_funct()


        def make_o_at_tgt_functs(o_idx):

            def _pred2subg_function(obs, goal):
                g_pos = self.get_o_goal_pos(goal, o_idx)
                # object_at_target is only true if laying on table.
                g_pos[2] = self.table_height
                subg = [o_idx+1] + list(g_pos)
                return subg

            def _obs2pred_function(obs, goal):
                tgt_pos = _pred2subg_function(obs, goal)[1:]
                o_pos = self.get_o_pos(obs, o_idx)
                distance = np.linalg.norm(o_pos - tgt_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        for o in range(self.n_objects):
            pred_name = 'o{}_at_target'.format(o)
            self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] = make_o_at_tgt_functs(o)

    # def obs2preds_single(self, obs, goal):
    #     preds = {}
    #     for p, obs2pred_func in self.obs2pred_functs.items():
    #         preds[p] = obs2pred_func(obs, goal.copy())
    #     preds = {p: int(v) for p, v in preds.items()}
    #     one_hot = np.array([preds[k] for k in sorted(preds.keys())])
    #     return preds, one_hot

    def get_o_pos(self, obs, o_idx):
        start_idx = (o_idx + 1) * 3
        end_idx = start_idx + 3
        o_pos = obs[start_idx:end_idx]
        return o_pos.copy()

    def get_o_goal_pos(self, goal, o_idx):
        start_idx = (o_idx + 1) * 3
        if self.gripper_has_target is False:
            start_idx -= 3
        end_idx = start_idx + 3
        g_pos = goal[start_idx:end_idx]
        return g_pos

    # def gen_plan_single(self, obs_preds, gripper_has_target, goal_preds):
    #
    #     domain, problem = self.gen_pddl_domain_problem(obs_preds, goal_preds, gripper_has_target=gripper_has_target)
    #
    #     planner = Propositional_Planner()
    #     plan_start = time.time()
    #     plan, state_history = planner.solve(domain, problem, return_states=True)
    #     duration = time.time() - plan_start
    #     if duration > 10.0:
    #         print("Plan generation took {:.2f} sec.".format(time.time() - plan_start))
    #
    #     plan_acts = []
    #     # world_states = [state_history[0]]
    #     goal_achieved = False
    #     if plan is None:
    #         print('No plan was found')
    #         with open('no_plan_dom.pddl', 'w') as f:
    #             f.write(domain)
    #         with open('no_plan_prob.pddl', 'w') as f:
    #             f.write(problem)
    #         goal_achieved = False
    #     elif plan == []:
    #         goal_achieved = True
    #         # print("Goal already achieved")
    #     else:
    #         # print('plan:')
    #         for i, act in enumerate(plan):
    #             plan_acts.append(act.name)
    #
    #     return plan_acts, state_history

    # def preds2subgoal_obs(self, preds, obs, goal):
    #     subgoal_obs = obs.copy()
    #     true_preds = [p for p,v in preds.items() if v == 1]
    #     iter_ctr = 0
    #     while True:
    #         new_subgoal_obs = subgoal_obs.copy()
    #         for p in true_preds:
    #             pred_subgoal = self.pred2subg_functs[p](new_subgoal_obs, goal)
    #             obs_start_idx = pred_subgoal[0] * 3
    #             new_subgoal_obs[obs_start_idx:obs_start_idx + 3] = pred_subgoal[1:]
    #
    #         if str(new_subgoal_obs) == str(subgoal_obs):
    #             break
    #         iter_ctr += 1
    #         if iter_ctr > len(true_preds):
    #             print("TODO: This should not happen. Check why this does not converge...")
    #             break
    #         subgoal_obs = new_subgoal_obs
    #
    #     return subgoal_obs

    def gen_actions(self, n_objects):
        actions = []
        not_grasped_str = ''
        for o in range(n_objects):
            not_grasped_str += '(not (grasped_o{}))'.format(o)
        move_gripper_to_o_act_template = "(:action move_gripper_to__o{} \n\t:parameters () \n\t:precondition () \n\t:effect (and (gripper_at_o{}) {} {} (not (gripper_at_target)) )\n)\n\n"
        move_o_to_target_template = "(:action move__o{}_to_target \n\t:parameters () \n\t:precondition (and (gripper_at_o{}) ) \n\t:effect (and (o{}_at_target) )\n)\n\n"
        move_o1_on_o2_act_template = "(:action move__o{}_on__o{}  \n\t:parameters () \n\t:precondition (and (gripper_at_o{}) ) \n\t:effect (and (o{}_on_o{})  {} )\n)\n\n"

        for o in range(n_objects):
            # Grasp object action
            not_o2_on_o_str = ''
            for o2 in range(n_objects):
                if o == o2:
                    continue
                not_o2_on_o_str += ' (not (o{}_on_o{}))'.format(o2, o)
            not_elsewhere_str = ''
            for o_other in range(n_objects):
                if o_other == o:
                    continue
                not_elsewhere_str += '(not (gripper_at_o{}))'.format(o_other)

            # Move gripper to object action
            move_gripper_to_o_act = move_gripper_to_o_act_template.format(o, o, not_elsewhere_str, not_o2_on_o_str)
            actions.append(move_gripper_to_o_act)

            # Move o to target action. This is to place the first object on the ground on which other objects will be stacked.
            move_o_to_target_act = move_o_to_target_template.format(o, o, o)
            actions.append(move_o_to_target_act)

            # Move o1 on o2 action. This is to stack objects on other objects.
            for o2 in range(n_objects):
                # if o-1 != o2:
                #     continue # To restrict the action space, only allow to put an object with number o onto o-1 (e.g. object 1 on object 0, but not object 0 on object 2.)
                not_o3_on_o2_str = ''
                for o3 in range(n_objects):
                    if o3 == o2:
                        continue
                    if o3 == o:
                        continue
                    not_o3_on_o2_str += ' (not (o{}_on_o{}))'.format(o3, o2)
                move_o1_on_o2_act = move_o1_on_o2_act_template.format(o, o2, o, o, o2, not_o3_on_o2_str)
                actions.append(move_o1_on_o2_act)

        not_elsewhere_str = ''
        for o in range(n_objects):
            not_elsewhere_str += '(not (gripper_at_o{}))'.format(o)
        move_gripper_to_target = "(:action move_gripper_to_target \n\t:parameters () \n\t:precondition (and {}) \n\t:effect (and (gripper_at_target) {})\n)\n\n".format(
            not_grasped_str, not_elsewhere_str)

        actions.append(move_gripper_to_target)
        return actions

    def gen_pddl_domain_problem(self, obs_preds, goal_preds, gripper_has_target=True):

        head = "(define (domain tower) (:requirements :strips) \n{})\n"

        predicates = "(:predicates \n"
        for p in sorted(obs_preds.keys()):
            predicates += "\t{}\n".format(p)
        predicates += ")\n"

        n_objects = len([p for p in obs_preds if p.find("gripper_at_o") != -1])

        actions = self.gen_actions(n_objects)
        body = predicates + "".join(actions)
        domain = head.format(body)

        # Define problem
        problem = "(define (problem pb1) (:domain tower)\n {}\n)\n"

        # Define initial state
        init_str = "(:init\n"
        for pred in obs_preds:
            if obs_preds[pred] == 1:
                init_str += "\t ({})\n".format(pred)
        init_str += ")\n"

        # Define goal
        goal_str = "(:goal (and \n"
        true_goal_preds_list = [g for g, v in goal_preds.items() if v == 1]
        for g in true_goal_preds_list:
            goal_str += '\t ({})\n'.format(g)
        goal_str += '))\n\n'

        problem = problem.format(init_str + goal_str)

        return domain, problem

