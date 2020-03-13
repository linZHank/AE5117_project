"""
Task environment for cablers cooperatively control a device to track and catch a movable target.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi
from numpy import random
import matplotlib
import matplotlib.pyplot as plt


class CablerEnv(object):
    """
    Cabler env class
    """
    def __init__(self):
        self.world_radius = 1
        self.catcher = np.zeros(2)
        self.target = self.catcher + np.array([0,self.world_radius/2])
        self.cabler_1 = np.array([self.world_radius*np.cos(pi/6),self.world_radius*np.sin(pi/6)])
        self.cabler_2 = np.array([self.world_radius*np.cos(5*pi/6),self.world_radius*np.sin(5*pi/6)])
        self.cabler_3 = np.array([self.world_radius*np.cos(-pi/2),self.world_radius*np.sin(-pi/2)])

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        # plot world boundary and
        bound = plt.Circle((0,0), self.world_radius, linewidth=2, color='k', fill=False)
        fig, ax = plt.subplots()
        ax.add_artist(bound)
        # draw objects
        plt.scatter(self.cabler_1[0], self.cabler_1[1], s=200, marker='p', color='crimson')
        plt.scatter(self.cabler_2[0], self.cabler_2[1], s=200, marker='p', color='orangered')
        plt.scatter(self.cabler_3[0], self.cabler_3[1], s=200, marker='p', color='magenta')
        plt.scatter(self.catcher[0], self.catcher[1], s=100, marker='o', color='red')
        plt.scatter(self.target[0], self.target[1], s=400, marker='*', color='gold')
        # draw cables
        plt.plot([self.cabler_1[0],self.catcher[0]], [self.cabler_1[1],self.catcher[1]], linewidth=0.5, linestyle=':', color='k')
        plt.plot([self.cabler_2[0],self.catcher[0]], [self.cabler_2[1],self.catcher[1]], linewidth=0.5, linestyle=':', color='k')
        plt.plot([self.cabler_3[0],self.catcher[0]], [self.cabler_3[1],self.catcher[1]], linewidth=0.5, linestyle=':', color='k')
        # set axis
        plt.axis(1.1*np.array([-self.world_radius,self.world_radius,-self.world_radius,self.world_radius]))
        plt.show()

if __name__ == '__main__':
    env=CablerEnv()
    env.render()