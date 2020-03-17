#!/usr/bin/python3
import sys
import os
import numpy as np
from numpy import random
from datetime import datetime
import matplotlib.pyplot as plt

from envs.tri_cabler_kine_env import TriCablerKineEnv
from agents.q_table import QTableAgent


if __name__ == '__main__':
    env = TriCablerKineEnv()
    agent_0 = QTableAgent(name='cabler_0', env=env)
    agent_1 = QTableAgent(name='cabler_1', env=env)
    agent_2 = QTableAgent(name='cabler_2', env=env)

    num_episodes = 2000
    sedimentary_returns = []
    episodic_returns = []
    ep = 0
    while ep < num_episodes:
        done, total_reward = False, []
        obs, _ = env.reset()
        state_0, s_i_0 = agent_0.obs_to_state(obs)
        agent_0.linear_epsilon_decay(episode=ep, decay_period=1200)
        state_1, s_i_1 = agent_1.obs_to_state(obs)
        agent_1.linear_epsilon_decay(episode=ep, decay_period=1200)
        state_2, s_i_2 = agent_2.obs_to_state(obs)
        agent_2.linear_epsilon_decay(episode=ep, decay_period=1200)
        for i in range(180):
            a_i_0 = agent_0.epsilon_greedy(s_i_0)
            a_i_1 = agent_1.epsilon_greedy(s_i_1)
            a_i_2 = agent_2.epsilon_greedy(s_i_2)
            action = (env.cabler_0-env.catcher)/np.linalg.norm(env.cabler_0-env.catcher)*agent_0.actions[a_i_0] + (env.cabler_1-env.catcher)/np.linalg.norm(env.cabler_1-env.catcher)*agent_1.actions[a_i_1] + (env.cabler_2-env.catcher)/np.linalg.norm(env.cabler_2-env.catcher)*agent_2.actions[a_i_2]
            env.render()
            obs, rew, done, _ = env.step(action)
            next_state_0, ns_i_0 = agent_0.obs_to_state(obs)
            next_state_1, ns_i_1 = agent_1.obs_to_state(obs)
            next_state_2, ns_i_2 = agent_2.obs_to_state(obs)
            agent_0.train(s_i_0, a_i_0, ns_i_0, rew)
            agent_1.train(s_i_1, a_i_1, ns_i_0, rew)
            agent_2.train(s_i_2, a_i_2, ns_i_0, rew)

            total_reward.append(rew)
            print("episode: {}, step: {}, epsilon: {} \nobs: {} \nreward: {}".format(ep+1, i+1, agent_0.epsilon, obs, rew))
            state_0, s_i_0 = next_state_0, ns_i_0
            state_1, s_i_1 = next_state_1, ns_i_1
            state_2, s_i_2 = next_state_2, ns_i_2

            if done:
                break

        print("\n---episode: {}, total_reward: {} \n---\n".format(ep+1, sum(total_reward)))
        sed_return = (sum(total_reward)+sum(episodic_returns))/(ep+1)
        sedimentary_returns.append(sed_return)
        episodic_returns.append(sum(total_reward))

        ep += 1


    # save Q-tables
    save_dir = os.path.join(sys.path[0], 'saved_models/tri_cabler_kine/qtable', datetime.now().strftime("%Y-%m-%d-%H-%M"))
    agent_0.save_table(save_dir=save_dir)
    agent_1.save_table(save_dir=save_dir)
    agent_2.save_table(save_dir=save_dir)
    # plot total reward
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(sedimentary_returns))+1, sedimentary_returns)
    plt.show()
