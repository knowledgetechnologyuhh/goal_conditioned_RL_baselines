
from gym.envs.registration import register

# blockstack environment using the Fetch robot
IDsAndEPs = [['BlockStackMujocoEnv', 'stack_blocks:'], ['BlockPickAndPlaceMujocoEnv', 'pick_and_place:'],
             ['BlockSlideMujocoEnv', 'slide:'], ['BlockReachMujocoEnv', 'reach:'], ['BlockPushMujocoEnv', 'push:']]
for n_objects in range(0, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects,
                  'gripper_goal': gripper_goal}
        if gripper_goal == 'gripper_none' and n_objects == 0: # Disallow 0 objects and no gripper in goal, because this would zero the goal space size.
            continue
        max_ep_steps = 50 * (n_objects)
        max_ep_steps = max(50, max_ep_steps)
        for idep in IDsAndEPs:
            register(
                id=idep[0]+'-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
                entry_point='wtm_envs.mujoco.blocks.' + idep[1] + idep[0],
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,
        )

# causal dependencies environment using the Fetch robot
for n_objects in range(0, 5):
    kwargs = {'n_objects': n_objects}
    max_ep_steps = 50 * (n_objects)
    max_ep_steps = max(50, max_ep_steps)
    register(
        id='CausalDependenciesMujocoEnv-o{}-v0'.format(kwargs['n_objects']),
        entry_point='wtm_envs.mujoco.causal_dep.unlock_goal:CausalDependenciesMujocoEnv',
        kwargs=kwargs,
        max_episode_steps=max_ep_steps)

# blockstack environment using the Key robot
for n_objects in range(0, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects,
                  'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects)
        max_ep_steps = max(50, max_ep_steps)
        register(
            id='KeybotBlockStackMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects'],),
            entry_point='wtm_envs.mujoco.keybot.stack_blocks:KeybotBlockStackMujocoEnv',
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
            id='JarvisbotBlockStackMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.mujoco.jarvisbot.stack_blocks:JarvisbotBlockStackMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# blockstack environment using the Nico robot
for n_objects in range(0, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects)
        max_ep_steps = max(50, max_ep_steps)

        register(
            id='NicobotBlockStackMujocoEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.mujoco.nicobot.stack_blocks:NicobotBlockStackMujocoEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# Physical blockstack environment using the Key robot
for n_objects in range(0, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects)
        max_ep_steps = max(50, max_ep_steps)

        register(
            id='KeybotBlockStackPhysicalEnv-{}-o{}-v1'.format(kwargs['gripper_goal'], kwargs['n_objects']),
            entry_point='wtm_envs.physical.keybot.stack_blocks:KeybotBlockStackPhysicalEnv',
            kwargs=kwargs,
            max_episode_steps=max_ep_steps,
        )

# Rocker environment using the Fetch robot
for n_objects in range(0, 5):
    for gripper_goal in ['gripper_random', 'gripper_above', 'gripper_none']:
        kwargs = {'n_objects': n_objects, 'gripper_goal': gripper_goal}
        max_ep_steps = 50 * (n_objects)
        max_ep_steps = max(50, max_ep_steps)
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

# ReacherEnv using coppelia sim
for IK in [0, 1]:  # whether to use inverse kinematics
    kwargs = {'ik': IK}
    register(id='CopReacherEnv-ik{}-v0'.format(kwargs['ik']),
         entry_point='wtm_envs.coppelia.cop_reach_env:ReacherEnvHandler',
         kwargs=kwargs,
         max_episode_steps=200)

# UR5
for obs_type in range(0, 4):
    kwargs = {'obs_type': obs_type}
    register(id='UR5ReacherEnv-v{}'.format(kwargs['obs_type']),
             entry_point='wtm_envs.mujoco.ur5.reaching:Ur5ReacherEnv',
             kwargs=kwargs,
             max_episode_steps=100) # originally is 600 but it is too long
