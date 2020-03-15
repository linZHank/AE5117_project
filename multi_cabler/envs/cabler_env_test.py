#!/usr/bin/python3
from cabler_world_env import CablerEnv
from numpy import random


if __name__ == '__main__':
    env=CablerEnv()
    obs, info = env.reset()
    for st in range(200):
        env.step(random.randn(2))
        print("obs: {} \ninfo: {}".format(obs, info))
        env.render()
