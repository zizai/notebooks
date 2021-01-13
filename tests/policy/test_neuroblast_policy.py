import torch
from neuroblast.agents.utils import normalize_obs, tanh_to_action
from chaosbreaker.envs.robotics.cheetah import CheetahEnv
from neuroblast.policy.neuroblast_policy import NeuroblastA2C


def test_neuroblast_policy():
    env = CheetahEnv()
    model = NeuroblastA2C(env.observation_space, env.action_space)
    obs = env.reset()

    for i in range(100):
        obs = normalize_obs(obs, env.observation_space.low, env.observation_space.high)
        obs = torch.as_tensor(obs, dtype=torch.float)
        a, v, logp_a = model.step(obs)
        action = tanh_to_action(a, env.action_space.low, env.action_space.high)
        obs, _, _, _ = env.step(action)
        print(obs)
