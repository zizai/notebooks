import time

from chaosbreaker.envs.robotics import KukaDiverseObjectEnv


def main():
    num_steps = 1000

    env = KukaDiverseObjectEnv(renders=True)
    obs = env.reset()

    # env.render()

    while True:
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)
        time.sleep(0.01)


if __name__ == '__main__':
    main()
