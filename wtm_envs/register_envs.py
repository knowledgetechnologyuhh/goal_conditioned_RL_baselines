
from gym.envs.registration import register


for n_objects in range(0, 5):
    for min_tower_height in range(5):
        for max_tower_height in range(7):
            for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
                for reward_type in ['dense', 'sparse', 'subgoal']:
                    kwargs = {'reward_type': reward_type, 'n_objects': n_objects,
                              'gripper_goal': gripper_goal,
                              'min_tower_height': min_tower_height + 1,
                              'max_tower_height': max_tower_height + 1}
                    max_ep_steps = 50 * (n_objects)
                    max_ep_steps = max(50,max_ep_steps)

                    register(
                        id='BuildTowerMujocoEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.mujoco.tower.build_tower:BuildTowerMujocoEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )


# TODO (fabawi): register seperate environment for each task (More aligned to the original implementation)