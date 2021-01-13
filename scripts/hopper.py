import logging

import numpy as np
import time

from chaosbreaker.envs.locomotion.walker_env import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import Hopper


logger = logging.getLogger(__name__)


class HopperBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=True):
        self.robot = Hopper()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


def main():
    env = HopperBulletEnv()

    env.render()
    obs = env.reset()

    while 1:
        action = np.random.randn(3)
        obs = env.step(action)
        logger.info(obs)
        env.camera_adjust()
        time.sleep(1/100)


if __name__ == '__main__':
    main()
