import ray
from ray.tune import register_env

from neuroblast.agents import ESTrainer
from neuroblast.policy.default_policy import DefaultPolicy
from chaosbreaker.envs.robotics.cheetah import CheetahEnv


def env_creator(env_config):
    return CheetahEnv(**env_config)


def test_es():
    ray.init()

    config = {
        'policy': DefaultPolicy,
        'num_workers': 2,
        'episodes_per_batch': 1,
        'train_batch_size': 1,
        'noise_size': 100000000,
        'env_config': {
            'max_num_steps': 100,
        },
        'model': {
            'sparse': False,
            's_dim': 128,
        }
    }
    register_env('my_env', env_creator)
    trainer = ESTrainer(config, 'my_env')

    res = dict()
    for i in range(5):
        res = trainer.train()

    assert res['training_iteration'] == 5
    assert res['timesteps_total'] == 2000
    assert res['info']['update_ratio'] > 0
