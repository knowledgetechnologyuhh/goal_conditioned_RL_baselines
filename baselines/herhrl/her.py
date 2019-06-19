import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, penalty_magnitude, use_penalty):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        Ts = episode_batch['steps'][:,0,0]
        episode_idx_Ts = Ts[episode_idxs]
        episode_t_samples = np.array([np.random.randint(ep_t) for ep_t in episode_idx_Ts])
        transitions = {key: episode_batch[key][episode_idxs, episode_t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        rnd_batch = np.random.uniform(size=batch_size)
        t_diff = episode_idx_Ts - episode_t_samples
        future_offset = rnd_batch * t_diff
        future_offset = future_offset.astype(int)

        future_t = (episode_t_samples + 1 + future_offset)[her_indexes]

        for ft, et, ep_len in zip(future_t, episode_t_samples[her_indexes], episode_idx_Ts[her_indexes]):
            assert ft <= ep_len, "Future index too high"
            assert et <= ep_len, "Episode index too high"
            assert et <= ft, "Episode index {} higher than future index {}:".format(et, ft)

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        # transitions['r'] = reward_fun(**reward_params)
        transitions['r'] = reward_fun(**reward_params)

        # if in HL:

        # penalties = np.reshape(transitions['p'], transitions['r'].shape)
        if not use_penalty:
            # transitions['r'] = reward_fun(**reward_params)
            if np.mean(transitions['p']) > 0:
                assert False, "this lowest level should not have any penalties"
        else:

            # HER for penalty
            transitions['u'][her_indexes] = transitions['ag'][her_indexes]
            transitions['p'][her_indexes] = 0
            penalties = np.reshape(transitions['p'], transitions['r'].shape)
            # print(penalties)
            idx = np.argwhere(np.isclose(penalties, 1.))
            transitions['r'] = np.zeros_like(transitions['r'])
            transitions['r'][idx] = -penalty_magnitude
        # print(use_penalty)
        # print(transitions['r'])
        # idx = np.argwhere(np.isclose(penalties, 1.))
        # transitions['r'][idx] *= penalty_magnitude  # test to replace this to use only penalty as reward for the
                                                    # high-level
        # transitions['r'][idx] -= penalty_magnitude

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
