import numpy as np
from wtm_envs.mujoco.pddl_env import PDDLEnv

class PDDLAntEnv(PDDLEnv):
    distance_threshold = 0.3

    area_width = 18.0
    area_height = 18.0
    area_center = (0.0, 0.0)
    area_left = (area_center[0] - area_width / 2)
    area_right = (area_center[0] + area_width / 2)
    area_top = (area_center[1] - area_height / 2)
    area_bottom = (area_center[1] + area_height / 2)
    n_rooms_x = 2
    n_rooms_y = 2
    room_width = area_width / n_rooms_x
    room_height = area_height / n_rooms_y
    # room centers
    room_centers = {}
    for room_x in range(n_rooms_x):
        this_center_x = area_left + (room_x * room_width) + (room_width / 2)
        for room_y in range(n_rooms_y):
            this_center_y = area_top + (room_y * room_height) + (room_height / 2)
            room_centers["{}_{}".format(room_x, room_y)] = [this_center_x, this_center_y]

    room_connections = {}
    # room connection points (doors)
    for room_x in range(n_rooms_x):
        for room_y in range(n_rooms_y):
            for room_x2 in range(n_rooms_x):
                for room_y2 in range(n_rooms_y):
                    if room_x2 == room_x + 1 and room_y == room_y2:
                        this_connection_y = room_centers["{}_{}".format(room_x, room_y)][1]
                        this_connection_x = room_centers["{}_{}".format(room_x, room_y)][0] + (room_width / 2)
                        room_connections["{}_{}_{}_{}".format(room_x, room_y, room_x2, room_y2)] \
                            = [this_connection_x, this_connection_y]
                    if room_y2 == room_y + 1 and room_x == room_x2:
                        this_connection_y = room_centers["{}_{}".format(room_x, room_y)][1] + (room_height / 2)
                        this_connection_x = room_centers["{}_{}".format(room_x, room_y)][0]
                        room_connections["{}_{}_{}_{}".format(room_x, room_y, room_x2, room_y2)] \
                            = [this_connection_x, this_connection_y]

    def __init__(self):
        PDDLEnv.__init__(self)

    def _gen_pred_functs(self):
        self.pred2subg_functs = {}
        self.obs2pred_functs = {}

        # agent location in center of room
        def make_ag_at_room_center_functs(room_x, room_y):
            def _pred2subg_function(obs, goal):
                ag_tgt_xy = self.room_centers["{}_{}".format(room_x, room_y)].copy()
                ag_tgt_z = self.get_ag_pos(obs)[2]
                subg = [0] + list(ag_tgt_xy) + [ag_tgt_z]
                return subg

            def _obs2pred_function(obs, goal):
                ag_tgt_pos = _pred2subg_function(obs, goal)[1:]
                ag_pos = self.get_ag_pos(obs)
                distance = np.linalg.norm(ag_tgt_pos - ag_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        # agent location at door
        def make_ag_at_door_functs(room_x, room_y, room_x2, room_y2):
            assert "{}_{}_{}_{}".format(room_x, room_y, room_x2, room_y2) in self.room_connections.keys(), "Error, invalid room connection indices."
            def _pred2subg_function(obs, goal):
                ag_tgt_xy = self.room_connections["{}_{}_{}_{}".format(room_x, room_y, room_x2, room_y2)].copy()
                ag_tgt_z = self.get_ag_pos(obs)[2]
                subg = [0] + list(ag_tgt_xy) + [ag_tgt_z]
                return subg

            def _obs2pred_function(obs, goal):
                ag_tgt_pos = _pred2subg_function(obs, goal)[1:]
                ag_pos = self.get_ag_pos(obs)
                distance = np.linalg.norm(ag_tgt_pos - ag_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        # agent location in room
        def make_ag_in_room_functs(room_x, room_y):

            def _pred2subg_function(obs, goal):
                ## This predicate cannot be an action effect and, therefore does not need to provide a subgoal.
                return None

            def _obs2pred_function(obs, goal):
                closest_room = [0,0,1e15]
                ag_pos = self.get_ag_pos(obs)[:2]
                for oth_room_x in range(self.n_rooms_x):
                    for oth_room_y in range(self.n_rooms_y):
                        dist_to_center = np.linalg.norm(np.array(self.room_centers["{}_{}".format(oth_room_x, oth_room_y)]) - ag_pos)
                        if closest_room[2] > dist_to_center:
                            closest_room = [oth_room_x, oth_room_y, dist_to_center]
                if closest_room[0] == room_x and closest_room[1] == room_y:
                    return True
                else:
                    return False

            return _pred2subg_function, _obs2pred_function

        def make_tgt_in_room_functs(room_x, room_y):
            def _pred2subg_function(obs, goal):
                ## This predicate cannot be an action effect and, therefore does not need to provide a subgoal.
                return None

            def _obs2pred_function(obs, goal):
                closest_room = [0,0,1e15]
                tgt_pos = goal[:2]
                for oth_room_x in range(self.n_rooms_x):
                    for oth_room_y in range(self.n_rooms_y):
                        dist_to_center = np.linalg.norm(np.array(self.room_centers["{}_{}".format(oth_room_x, oth_room_y)]) - tgt_pos)
                        if closest_room[2] > dist_to_center:
                            closest_room = [oth_room_x, oth_room_y, dist_to_center]
                if closest_room[0] == room_x and closest_room[1] == room_y:
                    return True
                else:
                    return False

            return _pred2subg_function, _obs2pred_function

        def make_ag_at_target_functs():
            def _pred2subg_function(obs, goal):
                ag_tgt_pos = goal.copy()
                subg = [0] + list(ag_tgt_pos)
                return subg

            def _obs2pred_function(obs, goal):
                ag_pos = self.get_ag_pos(obs)
                tgt_pos = _pred2subg_function(obs, goal)[1:]
                distance = np.linalg.norm(tgt_pos - ag_pos)
                is_true = distance < self.distance_threshold
                return is_true

            return _pred2subg_function, _obs2pred_function

        pred_name = "ag_at_target"
        self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] \
            = make_ag_at_target_functs()

        for room_x in range(self.n_rooms_x):
            for room_y in range(self.n_rooms_y):
                pred_name = "ag_at_room_center_{}_{}".format(room_x, room_y)
                self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] \
                    = make_ag_at_room_center_functs(room_x, room_y)

                pred_name = "ag_in_room_{}_{}".format(room_x, room_y)
                self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] \
                    = make_ag_in_room_functs(room_x, room_y)

                pred_name = "target_in_room_{}_{}".format(room_x, room_y)
                self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] \
                    = make_tgt_in_room_functs(room_x, room_y)

                for room_x2 in range(self.n_rooms_x):
                    for room_y2 in range(self.n_rooms_y):
                        if (room_x2 == room_x + 1 and room_y == room_y2) or \
                           (room_y2 == room_y + 1 and room_x == room_x2):
                            pred_name = "ag_at_door_{}_{}_{}_{}".format(room_x, room_y, room_x2, room_y2)
                            self.pred2subg_functs[pred_name], self.obs2pred_functs[pred_name] \
                                = make_ag_at_door_functs(room_x, room_y, room_x2, room_y2)





    def gen_actions(self):

        actions = []
        move_to_door_act_template = "(:action move_to_door_{}_from_{} \n\t" \
                                    ":parameters () \n\t" \
                                    ":precondition (ag_at_room_center_{}) \n\t" \
                                    ":effect (and (ag_at_door_{}) (not (ag_at_room_center_{})) )\n)\n\n"

        move_to_room_center_act_template = "(:action move_to_room_center_{} \n\t" \
                                           ":parameters () \n\t" \
                                           ":precondition (and (ag_in_room_{}) {})\n\t" \
                                           ":effect (ag_at_room_center_{})\n)\n\n"

        move_to_room_center_from_door_act_template = "(:action move_to_room_center_{}_from_door_{} \n\t" \
                                           ":parameters () \n\t" \
                                           ":precondition (ag_at_door_{}) \n\t" \
                                           ":effect (and (ag_at_room_center_{}) (not (ag_at_door_{})) )\n)\n\n"

        move_to_target_act_template = "(:action move_to_target_in_room_{} \n\t" \
                                           ":parameters () \n\t" \
                                           ":precondition (and (ag_at_room_center_{}) (target_in_room_{}) ) \n\t" \
                                           ":effect (and (ag_at_target) (ag_in_room_{}) )\n)\n\n"

        for room_x in range(self.n_rooms_x):
            for room_y in range(self.n_rooms_y):
                room_str = "{}_{}".format(room_x, room_y)
                not_at_door_str = ""
                for room_x2 in range(self.n_rooms_x):
                    for room_y2 in range(self.n_rooms_y):
                        if not (room_x2 < self.n_rooms_x and room_y2 < self.n_rooms_y):
                            continue
                        if (room_x2 == room_x + 1 and room_y == room_y2) or (
                                room_y2 == room_y + 1 and room_x == room_x2):
                            door_str = "{}_{}_{}_{}".format(room_x, room_y, room_x2, room_y2)

                            move_to_room_center_from_door_act = move_to_room_center_from_door_act_template.format(
                                room_str, door_str, door_str, room_str, door_str)
                            actions.append(move_to_room_center_from_door_act)

                            room_str2 = "{}_{}".format(room_x2, room_y2)
                            move_to_room_center_from_door_act = move_to_room_center_from_door_act_template.format(
                                room_str2, door_str, door_str, room_str2, door_str)
                            actions.append(move_to_room_center_from_door_act)

                            move_to_door_act = move_to_door_act_template.format(door_str, room_str, room_str, door_str, room_str)
                            actions.append(move_to_door_act)
                            not_at_door_str += " (not (ag_at_door_{}))".format(door_str)

                        if (room_x2 == room_x - 1 and room_y == room_y2) or (
                                room_y2 == room_y - 1 and room_x == room_x2):
                            door_str = "{}_{}_{}_{}".format(room_x2, room_y2, room_x, room_y)
                            move_to_door_act = move_to_door_act_template.format(door_str, room_str, room_str, door_str, room_str)
                            actions.append(move_to_door_act)
                            not_at_door_str += " (not (ag_at_door_{}))".format(door_str)


                move_to_room_center_act = move_to_room_center_act_template.format(room_str, room_str, not_at_door_str, room_str)
                actions.append(move_to_room_center_act)

                move_to_target_act = move_to_target_act_template.format(room_str, room_str, room_str, room_str)
                actions.append(move_to_target_act)

        return actions


    def get_ag_pos(self, obs):
        return obs[:3]

