import numpy as np
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

    def gen_actions(self):
        n_objects = self.n_objects
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



