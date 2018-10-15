# TODO (fabawi): under construction
def sample_mask():
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
    # print("Enter mask str")
    # goldilocks_mask_idx = input()

mask = [int(n) for n in list(goldilocks_mask_idx)]

# print("mask: {}".format(mask))

return np.array(mask)
