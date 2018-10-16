# TODO (fabawi): under construction
import numpy as np
import os
import random

from baselines import logger


class CGM(object):

    def __init__(self, goal_size, **kwargs):
        self.goal_size = goal_size
        gm_successes = dict.fromkeys(["".join(["0" for _ in range(self.goal_size)])], [])
        subgoal_success = [[] for _ in range(self.goal_size)]
        self.params = {'gm_successes': gm_successes,
                       'subgoal_successes': subgoal_success,
                       'goldilocks_sampling': 'stochastic3_0.5_5_50_5' # TODO (fabawi): change to none
                       }

    def compute_successes(self, is_success_func, achieved_goal, desired_goal):
        # TODO (fabawi): implement!
        pass

    def update_params(self, params):
        self.params.update(params)

    def sample_mask(self):
        sorted_keys = sorted(self.params['gm_successes'].keys())
        avg_gm_success_rate = {}
        avg_dist_from_tgt_rate = {}
        tgt_success_rate = float(self.params['goldilocks_sampling'].split("_")[1]) / 100
        # sigma = 0.05
        # rnd_tgt_succ_rate = np.random.normal(tgt_success_rate, sigma)

        subgoal_success = self.params['subgoal_successes']
        # Default slot rate is such that the product of all slot rates is the target success rate.
        default_slot_rate = pow(tgt_success_rate, (1.0 / self.goal_size))
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
        p = float(self.params['goldilocks_sampling'].split("_")[3])

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
        n = float(self.params['goldilocks_sampling'].split("_")[2])
        rnd = random.randint(1, 100)
        if rnd < n:
            # goldilocks_mask_idx = '111111'
            goldilocks_mask_idx = "1" * self.goal_size
        else:
            goldilocks_mask_idx = np.random.choice(sorted_keys, p=norm_prob_sampling_list)

        mask = [int(n) for n in list(goldilocks_mask_idx)]

        return np.array(mask)
