import numpy as np

from neuroblast.policy.default_policy import DefaultPolicy, DenseModel, SparseModel
from chaosbreaker.envs.robotics.cheetah import CheetahEnv

env = CheetahEnv()
o_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]


def test_dense_model():
    s_dim = 512
    model = DenseModel(o_dim, s_dim, a_dim)

    obs = np.array(env.reset())
    a = model.forward(obs)

    assert a.shape == (a_dim,)

    vec = model.get_weights()
    vec = vec + 1

    model.set_weights(vec)

    assert np.equal(model.get_weights(), vec).all()


def test_sparse_model():
    s_dim = 1024
    model = SparseModel(o_dim, s_dim, a_dim)

    obs = np.array(env.reset()).reshape(1, -1)
    a = model.forward(obs)

    assert a.shape == (1, a_dim)

    vec = model.get_weights()
    vec = vec + 1

    model.set_weights(vec)

    assert np.equal(model.get_weights(), vec).all()


def test_default_policy():
    obs = np.array(env.reset())

    pi = DefaultPolicy(env.observation_space, env.action_space)
    pi.compute_single_action(obs)
