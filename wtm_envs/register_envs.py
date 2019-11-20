
from gym.envs.registration import register

# blockstack environment using the Fetch robot
for n_objects in range(-1, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects + 1,
                  'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects +1)

        register(
            id='BlockStackMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.mujoco.blockstack.stack_blocks:BlockStackMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# causal dependencies environment using the Fetch robot
for n_objects in range(-1, 5):
    kwargs = {'n_objects': n_objects + 1}
    max_ep_steps = 50 * (n_objects +1)

    register(
        id='CausalDependenciesMujocoEnv-o{}-v0'.format(kwargs['n_objects']),
        entry_point='wtm_envs.mujoco.causal_dep.unlock_goal:CausalDependenciesMujocoEnv',
        kwargs=kwargs,
        max_episode_steps=max_ep_steps)

# blockstack environment using the Key robot
for n_objects in range(-1, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects + 1,
                  'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects +1)

        register(
            id='KeybotTowerBuildMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects'],),
            entry_point='wtm_envs.mujoco.keybot.build_tower:KeybotTowerBuildMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# blockstack environment using the Jarvis robot
for n_objects in range(0, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * n_objects
        max_ep_steps = max(50, max_ep_steps)

        register(
            id='JarvisbotTowerBuildMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.mujoco.jarvisbot.build_tower:JarvisbotTowerBuildMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# blockstack environment using the Nico robot
for n_objects in range(-1, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects + 1, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects + 1)

        register(
            id='NicobotTowerBuildMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.mujoco.nicobot.build_tower:NicobotTowerBuildMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# Physical blockstack environment using the Key robot
for n_objects in range(-1, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects + 1, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects +1)

        register(
            id='KeybotTowerBuildPhysicalEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.physical.keybot.build_tower:KeybotTowerBuildPhysicalEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# Rocker environment using the Fetch robot
for n_objects in range(-1, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects + 1, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects +1)

        register(
            id='RockerMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.mujoco.rocker.press_grasp:RockerMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# Hook environment using the Fetch robot
for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
    for easy in range(2):
        kwargs = {'gripper_goal': gripper_goal, 'easy': easy}
        max_ep_steps = 50 * 3  # (n_objects +1)

        register(
            id='HookMujocoEnv-{}-e{}-v1'.format(kwargs['gripper_goal'], kwargs['easy']),
            entry_point='wtm_envs.mujoco.hook.pull_object:HookMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )


# Ant Envs:

# Ant Four Rooms:
max_ep_steps = 700
register(id='AntFourRoomsEnv-v0',
         entry_point='wtm_envs.mujoco.ant_four_rooms.navigate:AntFourRoomsEnv',
         max_episode_steps=max_ep_steps)

register(id='AntReacherEnv-v0',
         entry_point='wtm_envs.mujoco.ant_reacher.reach:AntReacherEnv',
         max_episode_steps=max_ep_steps)

