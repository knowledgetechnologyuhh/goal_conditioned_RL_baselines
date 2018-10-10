# from gym.envs.registration import register
# from wtm_envs.mujoco.fetch.build_tower import FetchBuildTowerEnv
#
#
# for n_objects in range(5):
#     for min_tower_height in range(5):
#         for max_tower_height in range(7):
#             for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
#                 for reward_type in ['dense', 'sparse', 'subgoal']:
#                     kwargs = {'reward_type': reward_type, 'n_objects': n_objects + 1,
#                               'gripper_goal': gripper_goal,
#                               'min_tower_height': min_tower_height + 1,
#                               'max_tower_height': max_tower_height + 1}
#                     max_ep_steps = 50 * (n_objects +1)
#
#                     register(
#                         id='FetchBuildTowerEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
#                                                                              kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
#                         entry_point='wtm_envs.mujoco.fetch.build_tower:FetchBuildTowerEnv',
#                         kwargs=kwargs,
#                         max_episode_steps=max_ep_steps,
#                     )
#

from gym.envs.registration import registry, register, make, spec
from wtm_envs.mujoco.fetch.build_tower import FetchBuildTowerEnv


for n_objects in range(5):
    for min_tower_height in range(5):
        for max_tower_height in range(7):
            for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
                for reward_type in ['dense', 'sparse', 'subgoal']:
                    kwargs = {'reward_type': reward_type, 'n_objects': n_objects + 1,
                              'gripper_goal': gripper_goal,
                              'min_tower_height': min_tower_height + 1,
                              'max_tower_height': max_tower_height + 1}
                    max_ep_steps = 50 * (n_objects +1)

                    register(
                        id='FetchBuildTowerEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.mujoco.fetch.build_tower:FetchBuildTowerEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )


