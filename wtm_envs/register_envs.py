
from gym.envs.registration import register

# Tower environment using the Fetch robot
for n_objects in range(-1, 5):
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
                        id='TowerBuildMujocoEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.mujoco.tower.build_tower:TowerBuildMujocoEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )

for n_objects in range(-1, 5):
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
                        id='KeybotTowerBuildMujocoEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.mujoco.keybot.build_tower:KeybotTowerBuildMujocoEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )

for n_objects in range(-1, 5):
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
                        id='JarvisbotTowerBuildMujocoEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.mujoco.jarvisbot.build_tower:JarvisbotTowerBuildMujocoEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )

for n_objects in range(-1, 5):
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
                        id='NicobotTowerBuildMujocoEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.mujoco.nicobot.build_tower:NicobotTowerBuildMujocoEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )

for n_objects in range(-1, 5):
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
                        id='KeybotTowerBuildPhysicalEnv-{}-{}-o{}-h{}-{}-v1'.format(kwargs['reward_type'], kwargs['gripper_goal'],
                                                                             kwargs['n_objects'], kwargs['min_tower_height'], kwargs['max_tower_height']),
                        entry_point='wtm_envs.physical.keybot.build_tower:KeybotTowerBuildPhysicalEnv',
                        kwargs=kwargs,
                        max_episode_steps=max_ep_steps,
                    )

# TODO (fabawi): register seperate environment for each task (More aligned to the original implementation)