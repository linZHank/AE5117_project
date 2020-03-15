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


class CablerEnv(object):
    """
    Cabler env class
    """
    def __init__(self):
        # fixed
        self.world_radius = 1.
        self.max_steps = 200
        self.rate = 30
        self.fig, self.ax = plt.subplots()
        # variable, we use polar coord to represent objects location
        self.catcher = np.zeros(2)
        self.target = self.catcher + np.array([pi/2, self.world_radius/2])
        self.cabler_1 = np.array([self.world_radius, pi/6])
        self.cabler_2 = np.array([self.world_radius, 5*pi/6])
        self.cabler_3 = np.array([self.world_radius, -pi/2])
        self.step_count = 0

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: {target_loc: array([x,y]), catcher_loc: array([x,y])
            info: 'coordinate type'
        """
        self.step_count = 0
        rho_t = random.uniform(0,self.world_radius)
        theta_t = random.uniform(-pi,pi)
        self.target = np.array([rho_t*np.cos(theta_t),rho_t*np.sin(theta_t)])
        rho_c = random.uniform(0,self.world_radius)
        theta_c = random.uniform(-pi,pi)
        self.catcher = np.array([rho_c*np.cos(theta_c),rho_c*np.sin(theta_c)])
        while np.linalg.norm(self.target-self.catcher) <= 0.05:
            rho_c = random.uniform(0,self.world_radius)
            theta_c = random.uniform(-pi,pi)
            self.catcher = np.array([rho_c*np.cos(theta_c),rho_c*np.sin(theta_c)])
        obs=dict(target=self.target, catcher=self.catcher)
        info='cartesian'

        return obs, info

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
        action = np.clip(action,-self.world_radius/20.,self.world_radius/20.)
        if np.linalg.norm(self.catcher+action) < self.world_radius:
            self.catcher += action
        self.step_count += 1
        obs=dict(target=self.target, catcher=self.catcher)
        reward = -np.linalg.norm(self.target-self.catcher)
        done = False
        if self.step_count >= self.max_steps:
            done = True
        info = 'cartesian'

        return obs, reward, done, info

    def render(self):
        # plot world boundary and
        bound = plt.Circle((0,0), self.world_radius, linewidth=2, color='k', fill=False)
        fig, ax = plt.gcf(), plt.gca()
        ax.add_artist(bound)
        # draw objects
        plt.scatter(self._polar_to_cartesian(self.cabler_1)[0], self._polar_to_cartesian(self.cabler_1)[1], s=200, marker='p', color='crimson')
        plt.scatter(self._polar_to_cartesian(self.cabler_2)[0], self._polar_to_cartesian(self.cabler_2)[1], s=200, marker='p', color='orangered')
        plt.scatter(self._polar_to_cartesian(self.cabler_3)[0], self._polar_to_cartesian(self.cabler_3)[1], s=200, marker='p', color='magenta')
        plt.scatter(self._polar_to_cartesian(self.catcher)[0], self._polar_to_cartesian(self.catcher)[1], s=100, marker='o', color='red')
        plt.scatter(self._polar_to_cartesian(self.target)[0], self._polar_to_cartesian(self.target)[1], s=400, marker='*', color='gold')
        # draw cables
        plt.plot([self._polar_to_cartesian(self.cabler_1)[0],self._polar_to_cartesian(self.catcher)[0]], [self._polar_to_cartesian(self.cabler_1)[1],self._polar_to_cartesian(self.catcher)[1]], linewidth=0.5, linestyle=':', color='k')
        plt.plot([self._polar_to_cartesian(self.cabler_2)[0],self._polar_to_cartesian(self.catcher)[0]], [self._polar_to_cartesian(self.cabler_2)[1],self._polar_to_cartesian(self.catcher)[1]], linewidth=0.5, linestyle=':', color='k')
        plt.plot([self._polar_to_cartesian(self.cabler_3)[0],self._polar_to_cartesian(self.catcher)[0]], [self._polar_to_cartesian(self.cabler_3)[1],self._polar_to_cartesian(self.catcher)[1]], linewidth=0.5, linestyle=':', color='k')
        # set axis
        plt.axis(1.1*np.array([-self.world_radius,self.world_radius,-self.world_radius,self.world_radius]))

        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()


    def _polar_to_cartesian(self, polar_coord):
        """
        Args:
            polar_coord: array([rho, theta])
        Returns:
            cart_coord: array([x, y])
        """
        cart_coord = np.array([polar_coord[0]*np.cos(polar_coord[1]), polar_coord[0]*np.sin(polar_coord[1])])

        return cart_coord

# if __name__ == '__main__':
#     env=CablerEnv()
#     obs, info = env.reset()
#     for st in range(200):
#         env.step(random.randn(2))
#         print("obs: {} \ninfo: {}".format(obs, info))
#         env.render()
