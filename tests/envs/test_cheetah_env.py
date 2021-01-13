import numpy
from neuroblast.agents.utils import normalize_obs, tanh_to_action
from chaosbreaker.envs.robotics.cheetah import CheetahEnv


def test_env():
    env = CheetahEnv(max_num_steps=500)
    env.reset()
    print(env.action_space, env.observation_space)
    assert not numpy.isinf(env.observation_space.low).any()
    assert not numpy.isinf(env.observation_space.high).any()
    assert not numpy.isinf(env.action_space.low).any()
    assert not numpy.isinf(env.action_space.high).any()
    action_dim = env.action_space.shape[0]

    done = False
    while not done:
        a = numpy.random.randn(action_dim)
        action = tanh_to_action(a, env.action_space.low, env.action_space.high)
        obs, rew, done, _ = env.step(action)
        print(rew)
