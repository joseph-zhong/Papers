#!/usr/local/bin/python

"""
main.py

Tutorial taken from Pytorch
Author: Adam Paszke
---

This tutorial shows PyTorch usage to train DQN on CartPole-v0
"""


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped

Tensor = FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTEnsor = torch.ByteTensor

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')






