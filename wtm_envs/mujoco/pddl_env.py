import numpy as np
from baselines.her_pddl.pddl.propositional_planner import Propositional_Planner
import time


class PDDLEnv:

    def __init__(self):
        self._gen_pred_functs()
        self.planner = Propositional_Planner()
        self.preds = self.get_preds()
        self._gen_pddl_domain()

    def get_preds(self):
        obs = self._get_obs()
        preds = self.obs2preds_single(obs['observation'], obs['desired_goal'])[0].keys()
        return preds

    def obs2preds_single(self, obs, goal):
        preds = {}
        for p, obs2pred_func in self.obs2pred_functs.items():
            preds[p] = obs2pred_func(obs, goal.copy())
        preds = {p: int(v) for p, v in preds.items()}
        one_hot = np.array([preds[k] for k in sorted(preds.keys())])
        return preds, one_hot

    def gen_plan_single(self, obs_preds, goal_preds):

        problem = self.gen_pddl_problem(obs_preds, goal_preds)
        max_time = 120
        plan_start = time.time()
        plan, state_history = self.planner.solve(self.domain, problem, return_states=True, max_time=max_time)
        duration = time.time() - plan_start
        if duration > max_time:
            print("Aborted planning. Plan generation took {:.2f} sec.".format(time.time() - plan_start))

        plan_acts = []
        # world_states = [state_history[0]]
        goal_achieved = False
        if plan is None:
            print('No plan was found')
            with open('no_plan_dom.pddl', 'w') as f:
                f.write(self.domain)
            with open('no_plan_prob.pddl', 'w') as f:
                f.write(problem)
            goal_achieved = False
        elif plan == []:
            goal_achieved = True
            # print("Goal already achieved")
        else:
            # print('plan:')
            for i, act in enumerate(plan):
                plan_acts.append(act.name)

        return plan_acts, state_history

    def preds2subgoal_obs(self, preds, obs, goal):
        subgoal_obs = obs.copy()
        true_preds = [p for p,v in preds.items() if v == 1]
        iter_ctr = 0
        while True:
            new_subgoal_obs = subgoal_obs.copy()
            for p in true_preds:
                pred_subgoal = self.pred2subg_functs[p](new_subgoal_obs, goal)
                obs_start_idx = pred_subgoal[0] * 3
                new_subgoal_obs[obs_start_idx:obs_start_idx + 3] = pred_subgoal[1:]

            if str(new_subgoal_obs) == str(subgoal_obs):
                break
            iter_ctr += 1
            if iter_ctr > len(true_preds):
                print("TODO: This should not happen. Check why this does not converge...")
                break
            subgoal_obs = new_subgoal_obs

        return subgoal_obs

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
            plan, world_states = self.gen_plan_single(obs_preds, goal_preds)

            self.plan_cache[cache_key] = (plan, world_states)
            if len(self.plan_cache) % 50 == 0:
                print("Number of cached plans: {}".format(len(self.plan_cache)))
        if return_states:
            return plan, world_states
        else:
            return plan

    def _gen_pddl_domain(self):

        head = "(define (domain tower) (:requirements :strips) \n{})\n"

        predicates = "(:predicates \n"
        for p in sorted(self.preds):
            predicates += "\t{}\n".format(p)
        predicates += ")\n"

        actions = self.gen_actions()
        body = predicates + "".join(actions)
        self.domain = head.format(body)

    def gen_pddl_problem(self, obs_preds, goal_preds):

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

        return problem

    def preds2subgoal(self, preds):
        obs = self._get_obs()['observation']
        subgoal_obs = self.preds2subgoal_obs(preds, obs, self.final_goal)
        subg = self._obs2goal(subgoal_obs)
        return subg

    def gen_actions(self, **kwargs):
        pass

    def _gen_pred_functs(self):
        pass





