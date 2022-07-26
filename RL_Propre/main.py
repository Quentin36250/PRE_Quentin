from pickletools import float8
import RL_ADMM
import ADMM
from gym import Env
from gym.spaces import Discrete,Box,Tuple
import numpy as np
import random
from stable_baselines3 import A2C,PPO,DQN
from stable_baselines3.common.env_util import make_vec_env
import math
from re import M
#from turtle import color
from cvxpy.reductions import solvers
import os
import time
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
import pandapower as pp
import cvxpy as cp
from pandapower.plotting.plotly import simple_plotly, pf_res_plotly
from pandapower.plotting import simple_plot, create_bus_collection
import multiprocessing as mp
from dataclasses import dataclass
import matplotlib.animation as anim
import csv
import ray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import reseau_elec
from parametre import*
from par_cvxpy import *
import sys
from stable_baselines3.common.callbacks import EvalCallback

num_cores=mp.cpu_count()
cp.installed_solvers()


if __name__=="__main__":
    G=reseau_elec.create_graph()
    env=RL_ADMM.ADMM_Env_rho_unique()
    eval_env=RL_ADMM.ADMM_Env_rho_unique()
    eval_callback=EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=50,
                             deterministic=True, render=False)
    states=env.observation_space.shape[0]
    print(states)
    actions=env.action_space.n
    #actions=env.action_space.shape[0]
    print(actions)
    env.reset()
    print(float(sys.argv[1]))
    model=DQN("MlpPolicy",env,verbose=1,learning_rate=float(sys.argv[1]),exploration_final_eps=0.1,buffer_size=300000)
    model.learn(total_timesteps=400,callback=eval_callback)
    model.save("test")
    print("fini")




