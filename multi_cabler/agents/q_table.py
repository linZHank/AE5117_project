#!/usr/bin/python3
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import random

class QTableAgent(object):
    def __init__(self, name, env):
        # fixed
        self.name = name
        self.world_radius = env.world_radius
        self.actions = np.array([0., self.world_radius/50])
        self.action_space = self.actions.shape
        self.d_box = np.array([[-np.inf, -self.world_radius],[-self.world_radius, -self.world_radius/10.],[-self.world_radius/10., -self.world_radius/50],[-self.world_radius/50,0],[0., self.world_radius/50.],[self.world_radius/50., self.world_radius/10.],[self.world_radius/10., self.world_radius],[self.world_radius, np.inf]]) # displaced ranges
        self.state_space = (self.d_box.shape[0],self.d_box.shape[0])
        # variable
        self.state = np.zeros((1,2))
        self.action = 0
        self.q_table = np.zeros([8,8,2]) # (1st state numbers, 2nd state numbers, ... , action numbers)
        # hyper-parameters
        self.epsilon = 1.
        self.init_eps = 1.
        self.final_eps = 0.1
        self.warmup_episodes = 10
        self.alpha = 0.01 # learning rate
        self.gamma = 0.99 # discount rate

    def epsilon_greedy(self, state_index):
        """
        Take action based on epsilon_greedy
        Args:
            state_index: [i_dx, i_dy]
        Returns:
            action_index:
        """
        if random.uniform() > self.epsilon:
            action_index = np.argmax(self.q_table[state_index[0],state_index[1]])
        else:
            action_index = random.randint(self.action_space[0])
            print("!{} Take a random action: {}:{}".format(self.name, action_index, self.actions[action_index]))

        return action_index

    def linear_epsilon_decay(self, episode, decay_period):
        """
        Returns the current epsilon for the agent's epsilon-greedy policy. This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et al., 2015). The schedule is as follows:
            Begin at 1. until warmup_steps steps have been taken; then Linearly decay epsilon from 1. to final_eps in decay_period steps; and then Use epsilon from there on.
        Args:
            decay_period: int
            episode: int
        Returns:
        """
        episodes_left = decay_period + self.warmup_episodes - episode
        bonus = (self.init_eps - self.final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., self.init_eps-self.final_eps)
        self.epsilon = self.final_eps + bonus

    def train(self, state_index, action_index, next_state_index, reward):
        """
        Update Q-table
        """
        self.q_table[state_index[0],state_index[1],action_index] = self.q_table[state_index[0],state_index[1],action_index]+self.alpha*(reward+self.gamma*np.max(self.q_table[next_state_index[0],next_state_index[1]])-self.q_table[state_index[0],state_index[1],action_index])


    def obs_to_state(self, obs):
        """
        Convert observation into indices in Q-table
        Args:
            obs: {target,catcher}
        Returns:
            state: array([dx, dy])
            state_index: [dim_0, dim_1, ...], index of state in Q-table
        """
        state = (obs['target']-obs['catcher']).reshape(1,-1) # array([dx, dy])
        # define state ranges
        # dx_box = np.array([[-np.inf, -self.world_radius],[-self.world_radius, -self.world_radius/20.],[-self.world_radius/20., 0],[0., self.world_radius/20.],[self.world_radius/20., self.world_radius],[self.world_radius, np.inf]]) # dx ranges
        # compute index of state in Q-table
        state_index = []
        for i, box in enumerate(self.d_box):
            if state[0,0] >= box[0] and state[0,0] < box[1]:
                state_index.append(i)
                break
        for i, box in enumerate(self.d_box):
            if state[0,1] >= box[0] and state[0,1] < box[1]:
                state_index.append(i)
                break

        return state, state_index



    def save_table(self):
        pass
