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
        # fixed
        self.world_radius = 1
        # variable, we use polar coord to represent objects location
        self.catcher = np.zeros(2)
        self.target = self.catcher + np.array([pi/2, self.world_radius/2])
        self.cabler_1 = np.array([self.world_radius, pi/6])
        self.cabler_2 = np.array([self.world_radius, 5*pi/6])
        self.cabler_3 = np.array([self.world_radius, -pi/2])

    def reset(self):
        """
        Reset targe and catcher to a random location
        Args:
        Return:
            obs: {target_loc: array([rho, theta]), catcher_loc: array([rho, theta])
            info: 'coordinate type'
        """
        self.target = np.array([random.uniform(0,self.world_radius), random.uniform(-pi,pi)])
        self.catcher = np.array([random.uniform(0,self.world_radius), random.uniform(-pi,pi)])
        while np.linalg.norm(self._polar_to_cartesian(self.target)-self._polar_to_cartesian(self.catcher)) <= 0.05:
            self.catcher = np.array([random.uniform(0,self.world_radius), random.uniform(-pi,pi)])
        obs=dict(target=self._polar_to_cartesian(self.target), catcher=self._polar_to_cartesian(self.catcher))
        info='cartesian'

        return obs, info

    def step(self, action):
        pass

    def render(self):
        # plot world boundary and
        bound = plt.Circle((0,0), self.world_radius, linewidth=2, color='k', fill=False)
        fig, ax = plt.subplots()
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
        plt.show()

    def _polar_to_cartesian(self, polar_coord):
        """
        Args:
            polar_coord: array([rho, theta])
        Returns:
            cart_coord: array([x, y])
        """
        cart_coord = np.array([polar_coord[0]*np.cos(polar_coord[1]), polar_coord[0]*np.sin(polar_coord[1])])

        return cart_coord

if __name__ == '__main__':
    env=CablerEnv()
    obs, info = env.reset()
    print("obs: {} \ninfo: {}".format(obs,info))
    env.render()
