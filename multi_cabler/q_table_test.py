#!/usr/bin/python3
import sys
import numpy as np

from envs.cabler_world_env import CablerEnv
from agents.q_table import QTableAgent



if __name__ == '__main__':
    env = CablerEnv()
    agent = QTableAgent(name='test', env=env)
    obs, _ = env.reset()
    state, i = agent.obs_to_state(obs)
    print("obs: {} \nstate: {} \nindex:{}".format(obs, state, i))
