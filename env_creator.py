from gym.spaces import Box
from gym.wrappers import TransformObservation
from ray.tune import register_env

from chaosbreaker.envs.locomotion.walker_env import Walker2DBulletEnv
from neuroblast.sim.cheetah import CheetahEnv
from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv, MinitaurPixEnv
from chaosbreaker.spaces import FloatBox, IntBox


def env_creator(env_cls, env_config):
    env = env_cls(**env_config)
    if isinstance(env.observation_space, (Box, FloatBox)):
        obs_min = env.observation_space.low
        obs_max = env.observation_space.high
        env = TransformObservation(env, lambda obs: 2 * (obs - obs_min) / (obs_max - obs_min) - 1)
    elif isinstance(env.observation_space, IntBox):
        obs_min = env.observation_space.low
        obs_max = env.observation_space.high
        env = TransformObservation(env, lambda obs: (obs - obs_min) / (obs_max - obs_min))
    return env


def register_envs():
    register_env("MinitaurEnv", lambda env_config: env_creator(MinitaurGymEnv, env_config))
    register_env("MinitaurPixEnv", lambda env_config: env_creator(MinitaurPixEnv, env_config))
    register_env("CheetahEnv", lambda env_config: env_creator(CheetahEnv, env_config))
    register_env("Walker2DEnv", lambda env_config: env_creator(Walker2DBulletEnv, env_config))
