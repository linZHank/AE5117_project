#!/usr/bin/python3
import sys
import numpy as np
from numpy import random

from envs.cabler_world_env import CablerEnv
from agents.q_table import QTableAgent


if __name__ == '__main__':
    env = CablerEnv()
    agent = QTableAgent(name='cabler_1', env=env)
    num_episodes = 100

    ep = 0
    while ep <= num_episodes:
        obs, _ = env.reset()
        state, s_i = agent.obs_to_state(obs)
        agent.linear_epsilon_decay(episode=ep, decay_period=40)
        for i in range(50):
            a_i = agent.epsilon_greedy(s_i)
            action = (env.cabler_1-env.catcher)/np.linalg.norm(env.cabler_1-env.catcher)*agent.actions[a_i]
            env.render()
            obs, rew, done, _ = env.step(action)
            next_state, ns_i = agent.obs_to_state(obs)
            agent.train(s_i, a_i, ns_i, rew)
            print("episode: {}, step: {}, epsilon: {} \nobs: {} \nstate: {} \nstate index:{} \nstate: {} \nreward: {}".format(ep+1, i+1, agent.epsilon, obs, state, s_i, next_state, rew))
            state, s_i = next_state, ns_i

            if done:
                break

        ep += 1
