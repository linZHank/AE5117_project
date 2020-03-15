#!/usr/bin/python3
import sys
import numpy as np
from numpy import random

from envs.cabler_world_env import CablerEnv
from agents.q_table import QTableAgent


if __name__ == '__main__':
    env = CablerEnv()
    agent = QTableAgent(name='test', env=env)
    obs, _ = env.reset()
    state, s_i = agent.obs_to_state(obs)
    for i in range(100):
        a_i = random.randint(2)
        action = (env.cabler_1-env.catcher)/np.linalg.norm(env.cabler_1-env.catcher)*agent.actions[a_i]
        env.render()
        obs, rew, done, _ = env.step(action)
        next_state, ns_i = agent.obs_to_state(obs)
        agent.train(s_i, a_i, ns_i, rew)
        print("step: {} \nobs: {} \nstate: {} \nstate index:{} \nstate: {} \nreward: {}".format(i+1, obs, state, s_i, next_state, rew))
