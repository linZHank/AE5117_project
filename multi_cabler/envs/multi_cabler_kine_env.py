#!/usr/bin/python3
"""
Task environment for cablers cooperatively control a device to track and catch a movable target.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi
from numpy import random
import time
import matplotlib
import matplotlib.pyplot as plt

import pdb


class MultiCablerKineEnv(object):
    """
    Multi cabler kinematics env class
    """
    def __init__(self, num_cablers=3):
        # fixed
        self.world_radius = 1.
        self.max_steps = 200
        self.rate = 30 # Hz
        # variable, we use polar coord to represent objects location
        self.catcher = dict(position=np.zeros(2), velocity=np.zeros(2))
        self.target = dict(position=np.zeros(2)+np.array([0,self.world_radius/2]), velocity=np.zeros(2))
        self.cablers = dict(
            id = ['cabler_'+str(i) for i in range(num_cablers)],
            position = np.transpose(np.array([self.world_radius*np.cos(np.arange(0,2*pi,2*pi/num_cablers)),self.world_radius*np.sin(np.arange(0,2*pi,2*pi/num_cablers))])),
            velocity = np.zeros([num_cablers,2])
        )
        # self.cabler_0 = np.array([self.world_radius*np.cos(pi/6),self.world_radius*np.sin(pi/6)])
        # self.cabler_1 = np.array([self.world_radius*np.cos(5*pi/6),self.world_radius*np.sin(5*pi/6)])
        # self.cabler_2 = np.array([self.world_radius*np.cos(-pi/2),self.world_radius*np.sin(-pi/2)])
        self.step_count = 0

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: {target_loc: array([x,y]), catcher_loc: array([x,y], cabler_loc: array([x,y]...)
            info: 'coordinate type'
        """
        pass

    def step(self, action):
        """
        Take a resolved velocity command
        Args:
            action: array([v_x,v_y])
        Returns:
            obs: {target_loc: array([x,y]), catcher_loc: array([x,y])
            reward:
            done: bool
            info: 'coordinate type'
        """
        pass

    def render(self):
        # plot world boundary and
        pass
