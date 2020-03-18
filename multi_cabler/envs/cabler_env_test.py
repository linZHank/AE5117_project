#!/usr/bin/python3
from multi_cabler_kine_env import MultiCablerKineEnv
from numpy import random


if __name__ == '__main__':
    env=MultiCablerKineEnv(num_cablers=5)
    env.render()
    # obs, info = env.reset()
    # for st in range(30):
    #     env.step(random.randn(2))
    #     print("obs: {} \ninfo: {}".format(obs, info))
    #     env.render()
