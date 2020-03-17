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

from tri_cabler_kine_env import TriCablerKineEnv

import pdb


class QuadCablerDynaEnv(TriCablerKineEnv):
    """
    Four cabler dynamics env class
    """
    def __init__(self):
        super().__init__()
