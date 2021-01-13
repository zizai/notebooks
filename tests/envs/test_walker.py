import numpy as np
import time

from chaosbreaker.envs.locomotion.walker_env import HopperBulletEnv


def test():
    env = HopperBulletEnv(render=False)

    env.render()
    obs = env.reset()

    while 1:
        action = np.random.randn(3)
        obs, rew, done, _ = env.step(action)
        print(obs)
        if done:
            break
