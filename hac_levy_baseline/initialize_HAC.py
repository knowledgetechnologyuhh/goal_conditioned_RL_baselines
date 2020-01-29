"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


# from design_agent_and_env import design_agent_and_env
import env_designs
from options import parse_options
from agent import Agent
from run_HAC import run_HAC
import importlib


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
FLAGS = parse_options()
env_import_name = "env_designs." + FLAGS.env+"_design_agent_and_env"
design_agent_and_env_module = importlib.import_module(env_import_name)

# Instantiate the agent and Mujoco environment.  The designer must assign values to the hyperparameters listed in the "design_agent_and_env.py" file.
agent, env = design_agent_and_env_module.design_agent_and_env(FLAGS)

# Begin training
run_HAC(FLAGS,env,agent)

