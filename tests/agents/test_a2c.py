import ray
import torch

from neuroblast.agents.a2c import A2CTrainer, A2CTorchPolicy, DEFAULT_CONFIG
from chaosbreaker.envs.robotics.cheetah import CheetahEnv
from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv
from neuroblast.utils.logger import setup_logger
from ray.tune import register_env

logger = setup_logger()


def test_a2c_policy():
    env = MinitaurGymEnv()
    obs = env.reset()
    policy = A2CTorchPolicy(env.observation_space, env.action_space, DEFAULT_CONFIG)
    a = torch.zeros(env.action_space.shape[0]).numpy()
    s = policy.get_initial_state()

    for i in range(100):
        a, s, _ = policy.compute_single_action(obs, prev_action=a, state=s)


def test_model_free():
    ray.init()

    config = {
        'use_dynamics': False,
        'num_workers': 2,
        'train_batch_size': 200,
        "train_every": 200,
    }
    register_env('MinitaurEnv', lambda env_config: MinitaurGymEnv(**env_config))
    trainer = A2CTrainer(config, 'MinitaurEnv')

    for i in range(5):
        res = trainer.train()
        logger.info(res)

    ray.shutdown()


def test_dynamics():
    ray.init()

    config = {
        'env': {
            'max_num_steps': 200,
        },
        'use_dynamics': True,
        'num_workers': 2,
        'plan_horizon': 10,
        'rollout_fragment_length': 50,
        "learning_starts": 200,
        "train_batch_size": 200,
        "train_every": 200,
        "num_sgd_iter": 3,
        "monitor": False
    }
    register_env('CheetahEnv', lambda env_config: CheetahEnv(**env_config))
    trainer = A2CTrainer(config, 'CheetahEnv')

    print(trainer.get_policy().num_params)

    for i in range(5):
        res = trainer.train()
        assert res['timesteps_this_iter'] == config['rollout_fragment_length'] * config['num_workers']
        logger.info(res['info'])

    print(trainer._logdir)

    ray.shutdown()
