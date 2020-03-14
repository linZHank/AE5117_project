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
        self.actions = np.array([0., env.world_radius/50]) # cmd_vel
        # variable
        self.state = np.zeros(2).reshape(1,-1)
        # hyper-parameters
        self.epsilon = 1.
        self.init_eps = 1.
        self.final_eps = 0.1
        self.warmup_episodes = 10
        self.alpha = 0.01 # learning rate
        self.gamma = 0.99 # discount rate

    def epsilon_greedy(self, state):
        """
        Take action based on epsilon_greedy
        Args:
            state: array([d_x, d_y])
        Returns:
            action: array([v_x, v_y])
        """
        if random.uniform() >self.epsilon:
            pass
        else:
            print("{} Take a random action!".format(self.name))

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

    def train(self):
        pass

    def obs_to_state(self, obs):
        """
        Convert observation into indices in Q-table
        Args:
            obs: {target,catcher}
        Returns:
            index: [dim_0, dim_1, ...]
        """
        pass

    def save_table(self):
        pass
