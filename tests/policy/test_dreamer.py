import torch
from neuroblast.agents.utils import normalize_obs, tanh_to_action
from chaosbreaker.envs.robotics.cheetah import CheetahEnv
from neuroblast.policy.dreamer import Dreamer


def test_dreamer():
    env = CheetahEnv()
    obs = env.reset()

    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    model = Dreamer(o_dim, a_dim)

    o = normalize_obs(obs, env.observation_space.low, env.observation_space.high)

    o = torch.as_tensor(o, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        a, pi, v, rew, state = model(o)
    action = tanh_to_action(a.squeeze().numpy(), env.action_space.low, env.action_space.high)

    env.step(action)
    print(action)
    print(pi)
    print(v, rew)
    print(state)
