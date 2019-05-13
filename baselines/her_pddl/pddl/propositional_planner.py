
from baselines.her_pddl.pddl.PDDL import PDDL_Parser
import time

class Propositional_Planner:

    #-----------------------------------------------
    # Solve
    #-----------------------------------------------

    def solve(self, domain, problem, return_states=False, max_time=None):
        # Parser
        parser = PDDL_Parser()
        parser.parse_domain(domain)
        parser.parse_problem(problem)
        # Parsed data
        actions = parser.actions
        state = parser.state
        initial_state = state.copy()
        goal_pos = parser.positive_goals
        goal_not = parser.negative_goals
        # Do nothing
        if self.applicable(state, goal_pos, goal_not):
            if return_states:
                return [], [initial_state]
            else:
                return []
        # Search
        visited = [state]
        fringe = [state, None]
        start = time.clock()
        while fringe:
            state = fringe.pop(0)
            plan = fringe.pop(0)
            for act in actions:
                if self.applicable(state, act.positive_preconditions, act.negative_preconditions):
                    new_state = self.apply(state, act.add_effects, act.del_effects)
                    if new_state not in visited:
                        # if self.applicable(new_state, goal_pos, goal_not):
                        now = time.clock()
                        dur = now - start
                        early_stop = dur > max_time
                        # This early stopping thing is ugly. Should be improved by using a better planning backend.
                        if self.applicable(new_state, goal_pos, goal_not) or early_stop:
                            full_plan = [act]
                            # full_state_history = [initial_state, new_state]
                            while plan:
                                act, plan = plan
                                full_plan.insert(0, act)
                                # full_state_history.insert(0, new_state)
                            if return_states:
                                full_state_history = [initial_state]
                                this_state = initial_state
                                for plan_act in full_plan:
                                    this_state = self.apply(this_state, plan_act.add_effects, plan_act.del_effects)
                                    full_state_history.append(this_state)
                                full_state_dict_history = []
                                for full_state in full_state_history:
                                    full_state_dict = {}
                                    for pred_name in parser.predicates:
                                        if [pred_name] in full_state:
                                            full_state_dict[pred_name] = 1
                                        else:
                                            full_state_dict[pred_name] = 0

                                    full_state_dict_history.append(full_state_dict)
                                return full_plan, full_state_dict_history
                            else:
                                return full_plan
                        visited.append(new_state)
                        fringe.append(new_state)
                        fringe.append((act, plan))
        if return_states:
            return None, None
        else:
            return None

    #-----------------------------------------------
    # Applicable
    #-----------------------------------------------

    def applicable(self, state, positive, negative):
        for i in positive:
            if i not in state:
                return False
        for i in negative:
            if i in state:
                return False
        return True

    #-----------------------------------------------
    # Apply
    #-----------------------------------------------

    def apply(self, state, positive, negative):
        new_state = []
        for i in state:
            if i not in negative:
                new_state.append(i)
        for i in positive:
            if i not in new_state:
              new_state.append(i)
        return new_state

# ==========================================
# Main
# ==========================================
if __name__ == '__main__':
    import sys
    domain = sys.argv[1]
    problem = sys.argv[2]
    planner = Propositional_Planner()
    plan = planner.solve(domain, problem)
    if plan:
        print('plan:')
        for act in plan:
            print(act)
    else:
        print('No plan was found')