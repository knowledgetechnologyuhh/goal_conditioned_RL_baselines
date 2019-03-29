n_episodes = 3
n_test_rollouts = 2
n_epochs = 2
rollout_batch_size = 2
num_cpu = 2



python3 experiment/train.py --n_eposides ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [] --policies_layers []

python3 experiment/train.py --n_eposides ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [7] --policies_layers [DDPG_HER_HRL_POLICY]

python3 experiment/train.py --n_eposides ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [7] --policies_layers [PDDL_POLICY]

python3 experiment/train.py --n_eposides ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.herhrl --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs} --n_subgoals_layers [7] --policies_layers [MIX_PDDL_HRL_POLICY]

python3 experiment/train.py --n_eposides ${n_episodes} --n_test_rollouts ${n_test_rollouts} --env TowerBuildMujocoEnv-sparse-gripper_random-o2-h1-1-v1 --algorithm baselines.her --render 0 --num_cpu ${num_cpu} --rollout_batch_size ${rollout_batch_size} --n_epochs ${n_epochs}