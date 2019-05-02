import numpy as np
from baselines.her_pddl.pddl.propositional_planner import Propositional_Planner
import time


class PDDLEnv:

    def __init__(self):
        self._gen_pred_functs()
        self.planner = Propositional_Planner()

    def _gen_pred_functs(self):
        self.pred2subg_functs = {}
        self.obs2pred_functs = {}


    def obs2preds_single(self, obs, goal):
        preds = {}
        for p, obs2pred_func in self.obs2pred_functs.items():
            preds[p] = obs2pred_func(obs, goal.copy())
        preds = {p: int(v) for p, v in preds.items()}
        one_hot = np.array([preds[k] for k in sorted(preds.keys())])
        return preds, one_hot


    def gen_plan_single(self, obs_preds, gripper_has_target, goal_preds):

        domain, problem = self.gen_pddl_domain_problem(obs_preds, goal_preds, gripper_has_target=gripper_has_target)


        plan_start = time.time()
        plan, state_history = self.planner.solve(domain, problem, return_states=True)
        duration = time.time() - plan_start
        if duration > 10.0:
            print("Plan generation took {:.2f} sec.".format(time.time() - plan_start))

        plan_acts = []
        # world_states = [state_history[0]]
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

    def gen_actions(self, **kwargs):
        pass


    def gen_pddl_domain_problem(self, obs_preds, goal_preds, **kwargs):
        pass


