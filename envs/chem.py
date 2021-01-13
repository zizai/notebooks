import random

import numpy as np
from collections import namedtuple
from ray import rllib

from neuroblast.datasets import ZINC250K, QM9
from torch_geometric.data import Data


class Chem(rllib.env.MultiAgentEnv):

    demos = None
    agent = None
    agent_id = None

    def __init__(self, env_config):
        self.root_dir = env_config['root_dir']
        self.read = []
        self.write = []

        self.demos = QM9(self.root_dir)

    def step(self, action):
        obs = random.sample(self.demos, 1)[0]
        reward = 0
        done = 1
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = random.sample(self.demos, 1)[0]
        return obs
