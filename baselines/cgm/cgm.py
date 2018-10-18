# TODO (fabawi): under construction
from collections import deque
import numpy as np
import os
import random

from baselines import logger


class CGM(object):

    def __init__(self, goal_size, curriculum_sampling, gripper_goal, exploit, **kwargs):
        self._goal_size = goal_size
        self._curriculum_sampling = curriculum_sampling
        self._gripper_goal = gripper_goal
        self._exploit = exploit

        if self._curriculum_sampling == 'none':
            glr_avg_hist_len = 10
        else:
            glr_avg_hist_len = int(self._curriculum_sampling.split("_")[-1])

        if self._curriculum_sampling != 'none':
            possible_goal_masks = ["".join(["1" for _ in range(self._goal_size)])]
        else:
            possible_goal_masks = []
        max_n = pow(2, self._goal_size)
        for n in range(max_n):
            mask_str = bin(n)[2:]
            mask_str = mask_str.rjust(self._goal_size, "0")
            possible_goal_masks.append(mask_str)
        self._gm_successes = dict.fromkeys(["".join(["0" for _ in range(self._goal_size)])], [])

        for mask_str in possible_goal_masks:
            self._gm_successes[mask_str] = deque(maxlen=glr_avg_hist_len)

        self._subgoal_successes = [deque(maxlen=glr_avg_hist_len) for _ in range(self._goal_size)]

        self._params = {'gm_successes': self._gm_successes,
                        'subgoal_successes': self._subgoal_successes,
                        'curriculum_sampling': 'none'
                        }

        self._mask = None

    def compute_successes(self, is_success_func, achieved_goal, desired_goal, observation, last_time_step=False):
        # Masking at observation
        inv_mask = 1 - self._mask
        goal = desired_goal * self._mask
        if self._gripper_goal == 'gripper_none':
            goal += (observation.copy()[:self._goal_size] * inv_mask)
        else:
            goal += (observation.copy()[3:self._goal_size + 3] * inv_mask)
        # Compute is_success
        is_success = is_success_func(achieved_goal * self._mask, desired_goal * self._mask)

        # Update the subgoals
        subgoal_success = [is_success_func(np.array([achieved_goal[i] * self._mask[i]]),
                                           np.array([desired_goal[i] * self._mask[i]]))
                           for i in range(len(desired_goal))]

        if last_time_step:
            [self._subgoal_successes[idx].append(subgoal_success[idx]) for idx in
             range(len(subgoal_success)) if self._mask[idx] == 1]
            goal_mask_str = ''.join(map(str, self._mask))
            self._gm_successes[goal_mask_str].append(is_success)
        # update the desired goal to the mask on observation
        desired_goal = goal

        return is_success, achieved_goal, desired_goal, observation

    def update_and_reset(self):
        # if self._exploit and ('none' in self._curriculum_sampling): # TODO (fabawi): in the custom_rollout there is a condition placed on when to update subgoals. I don't know if it's relevant here
        self._params.update({'subgoal_successes': self._subgoal_successes})

        self._params.update({'gm_successes': self._gm_successes,
                             'curriculum_sampling': self._curriculum_sampling
                             })

    def sample_mask(self):
        if self._params['curriculum_sampling'] == 'none':
            curriculum_mask_idx = "1" * self._goal_size
        elif self._params['curriculum_sampling'].find('stochastic3_') != -1:
            sorted_keys = sorted(self._params['gm_successes'].keys())
            avg_gm_success_rate = {}
            avg_dist_from_tgt_rate = {}
            tgt_success_rate = float(self._params['curriculum_sampling'].split("_")[1]) / 100
            # sigma = 0.05
            # rnd_tgt_succ_rate = np.random.normal(tgt_success_rate, sigma)

            subgoal_success = self._params['subgoal_successes']
            # Default slot rate is such that the product of all slot rates is the target success rate.
            default_slot_rate = pow(tgt_success_rate, (1.0 / self._goal_size))
            for gm in sorted_keys:
                avg_gm_success_rate[gm] = 1
                for idx, succ_hist in enumerate(subgoal_success):
                    if gm[idx] == "1":
                        if len(succ_hist) > 0:
                            avg_gm_success_rate[gm] *= np.mean(succ_hist)
                        else:
                            avg_gm_success_rate[gm] = default_slot_rate
                    else:
                        avg_gm_success_rate[gm] *= 1
                avg_dist_from_tgt_rate[gm] = abs(avg_gm_success_rate[gm] - tgt_success_rate)
            avg_dist_list = []
            for gm in sorted_keys:
                avg_dist_list.append(avg_dist_from_tgt_rate[gm])

            avg_dist_list = np.array(avg_dist_list)
            prob_sampling_list = 1 - avg_dist_list
            p = float(self._params['curriculum_sampling'].split("_")[3])

            # avg_dist_list = np.random.uniform(size=avg_dist_list.shape) # random distribution

            prob_sampling_list = np.power(prob_sampling_list, p)
            if np.sum(prob_sampling_list) == 0:
                prob_sampling_list = np.ones(prob_sampling_list.shape)
            norm_prob_sampling_list = prob_sampling_list / prob_sampling_list.sum(axis=0, keepdims=1)
            norm_prob_sampling_list = np.nan_to_num(norm_prob_sampling_list)
            sample_prob_fpath = logger.get_dir() + "/mask_sample_prob.csv"
            if not os.path.isfile(sample_prob_fpath):
                with open(sample_prob_fpath, "w") as f:
                    for k in sorted_keys:
                        f.write(k + " , ")
                    f.write("\n")
            with open(sample_prob_fpath, "a") as f:
                for p in norm_prob_sampling_list:
                    f.write(str(p) + ", ")
                f.write("\n")
            # Not use now:
            n = float(self._params['curriculum_sampling'].split("_")[2])
            rnd = random.randint(1, 100)
            if rnd < n:
                curriculum_mask_idx = "1" * self._goal_size
            else:
                curriculum_mask_idx = np.random.choice(sorted_keys, p=norm_prob_sampling_list)
        else:
            curriculum_mask_idx = "1" * self._goal_size
            print("Invalid curriculum sampling strategy {}".format(self._params['curriculum_sampling']))

        mask = [int(n) for n in list(curriculum_mask_idx)]
        self._mask = np.array(mask)
        return self._mask
