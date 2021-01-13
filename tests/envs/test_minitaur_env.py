import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from chaosbreaker.envs.robotics.minitaur.minitaur_env import MinitaurPixEnv
from neuroblast.agents.utils import tanh_to_action, normalize_obs
from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv


def test_env():
    env = MinitaurGymEnv(max_num_steps=100)
    env.reset()
    assert not np.isinf(env.observation_space.low).any()
    assert not np.isinf(env.observation_space.high).any()
    assert not np.isinf(env.action_space.low).any()
    assert not np.isinf(env.action_space.high).any()
    action_dim = env.action_space.shape[0]

    done = False
    i = 0
    while not done:
        a = np.random.randn(action_dim)
        action = tanh_to_action(a, env.action_space.low, env.action_space.high)
        obs, rew, done, _ = env.step(action)
        pix = env.render()
        # plt.imsave('filename_{}.png'.format(i), pix)
        # plt.imshow(pix)
        print(pix)
        i += 1


def test_pix_env():
    env = MinitaurPixEnv(max_num_steps=100)

    obs = env.reset()
    print(obs.min())
    assert env.observation_space.contains(obs)
    assert not np.isinf(env.observation_space.low).any()
    assert not np.isinf(env.observation_space.high).any()
    assert not np.isinf(env.action_space.low).any()
    assert not np.isinf(env.action_space.high).any()
    action_dim = env.action_space.shape[0]

    done = False
    i = 0
    print(env.observation_space)
    while not done:
        a = np.random.randn(action_dim)
        action = tanh_to_action(a, env.action_space.low, env.action_space.high)
        obs, rew, done, _ = env.step(action)
        # plt.imsave('filename_{}.png'.format(i), pix)
        # plt.imshow(pix)
        assert env.observation_space.contains(obs)
        i += 1
