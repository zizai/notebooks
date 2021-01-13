import logging
import time

from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


def env_creator(env_config):
    return HumanoidBulletEnv(**env_config)


def main():
    env = HumanoidBulletEnv(render=True)
    env.reset()

    for i in range(100):
        a = env.action_space.sample()

        obs, rew, _, _ = env.step(a)
        logger.info(rew)

        time.sleep(.01)


if __name__ == '__main__':
    main()
