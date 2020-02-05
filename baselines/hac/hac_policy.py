from baselines.util import (store_args)
from baselines.template.policy import Policy

from baselines.hac.agent import Agent
import baselines.hac.env_designs
from baselines.hac.options import parse_options
from baselines.hac.utils import EnvWrapper, check_envs
import os,sys,inspect
import importlib

class HACPolicy(Policy):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, polyak, batch_size,
            Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
            rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
            sample_transitions, gamma, reuse=False, **kwargs):
        Policy.__init__(self, input_dims, T, rollout_batch_size, **kwargs)

        # Determine training options specified by user.  The full list of available options can be found in "options.py" file.
        self.FLAGS = parse_options()
        self.FLAGS.mix_train_test = True
        self.FLAGS.retrain = True
        self.FLAGS.Q_values = True
        #  self.FLAGS.verbose = True
        #  self.FLAGS.penalty = True

        agent, env = self.init_levy(self.FLAGS)
        wtm_agent, wtm_env, self.FLAGS = self.wtm_env_levy_style(kwargs['make_env'], self.FLAGS)
        check_envs(env, wtm_env)

        self.original = False

        if self.original:
            self.agent = agent
            self.env = env
        else:
            self.agent = wtm_agent
            self.env = wtm_env


    def wtm_env_levy_style(self,make_env, FLAGS):
        env = make_env().env
        env = EnvWrapper(env, FLAGS, self.input_dims)
        # get modified FLAGS
        FLAGS = env.FLAGS

        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["subgoal_penalty"] = -FLAGS.time_scale
        agent_params["atomic_noise"] = [0.1 for i in range(8)]
        agent_params["subgoal_noise"] = [0.1 for i in range(len(env.sub_goal_thresholds))]
        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100
        FLAGS.id = 1
        agent = Agent(FLAGS,env,agent_params)

        return agent, env, FLAGS


    def init_levy(self, FLAGS):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir)
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        env_import_name = "baselines.hac.env_designs.ANT_FOUR_ROOMS_2_design_agent_and_env"
        design_agent_and_env_module = importlib.import_module(env_import_name)
        # simple tag for agent's tf scope
        FLAGS.id = 0
        # Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
        agent, env = design_agent_and_env_module.design_agent_and_env(FLAGS)
        return agent, env
