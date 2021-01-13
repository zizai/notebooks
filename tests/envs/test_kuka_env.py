import gym
import numpy
from chaosbreaker import spaces
from chaosbreaker.spaces import IntBox, Composite, FloatBox
from neuroblast.agents.utils import tanh_to_action, normalize_obs
from chaosbreaker.envs.robotics.kuka import KukaCompositeEnv, KukaGymEnv, KukaCompositeObservation


def test_env():
    env = KukaGymEnv(max_num_steps=100)
    env.reset()
    assert isinstance(env.action_space, FloatBox)
    assert isinstance(env.observation_space, FloatBox)

    assert not numpy.isinf(env.action_space.low).any()
    assert not numpy.isinf(env.action_space.high).any()

    assert not numpy.isinf(env.observation_space.low).any()
    assert not numpy.isinf(env.observation_space.high).any()

    action_dim = env.action_space.shape[0]
    done = False
    while not done:
        a = numpy.random.randn(action_dim)
        action = tanh_to_action(a, env.action_space.low, env.action_space.high)
        obs, rew, done, _ = env.step(action)
        o = normalize_obs(obs, env.observation_space.low, env.observation_space.high)


def test_composite_env():
    env = KukaCompositeEnv(max_num_steps=100)
    env.reset()
    assert isinstance(env.observation_space, Composite)
    assert isinstance(env.action_space, FloatBox)
    print(env.observation_space)
    print(env.action_space)

    assert not numpy.isinf(env.action_space.low).any()
    assert not numpy.isinf(env.action_space.high).any()

    print(env.observation_space.spaces, env.observation_space.names, env.observation_space.shape)
    for v in env.observation_space.spaces:
        assert not numpy.isinf(v.low).any()
        assert not numpy.isinf(v.high).any()

    action_dim = env.action_space.shape[0]
    done = False
    while not done:
        a = numpy.random.randn(action_dim)
        action = tanh_to_action(a, env.action_space.low, env.action_space.high)
        obs, rew, done, _ = env.step(action)
