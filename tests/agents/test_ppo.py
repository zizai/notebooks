import ray
from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv
from ray.tune import register_env

from neuroblast.agents.ppo.ppo import PPOTrainer
from chaosbreaker.envs.robotics.cheetah import CheetahEnv

ray.init()


def test_ppo_minitaur():
    config = {
        'sgd_minibatch_size': 64,
        'num_workers': 2,
        'num_sgd_iter': 10,
        'rollout_fragment_length': 50,
        'train_batch_size': 200,
    }

    register_env('MinitaurEnv', lambda env_config: MinitaurGymEnv(**env_config))
    trainer = PPOTrainer(config=config, env='MinitaurEnv')

    for i in range(3):
        result = trainer.train()
        print(result)

    for i in range(3):
        batch = trainer.evaluate()
        print(batch['rewards'].mean())


def test_ppo_cheetah():
    config = {
        'env_config': {
            'max_num_steps': 500,
        },
        'sgd_minibatch_size': 64,
        'num_workers': 2,
        'num_sgd_iter': 10,
        'rollout_fragment_length': 40,
        'train_batch_size': 400,
    }

    register_env('CheetahEnv', lambda env_config: CheetahEnv(**env_config))
    trainer = PPOTrainer(config=config, env='CheetahEnv')

    for i in range(5):
        result = trainer.train()
        print(result)

    for i in range(5):
        batch = trainer.evaluate()
        print(batch['rewards'].mean())


def test_ppo_neuroblast():
    config = {
        'env_config': {
            'max_num_steps': 500,
        },
        'sgd_minibatch_size': 64,
        'num_workers': 1,
        'num_sgd_iter': 10,
        'policy': 'neuroblast',
        'rollout_fragment_length': 40,
        'train_batch_size': 400,
    }

    register_env('CheetahEnv', lambda env_config: CheetahEnv(**env_config))
    trainer = PPOTrainer(config=config, env='CheetahEnv')

    for i in range(5):
        result = trainer.train()
        print(result)

    for i in range(5):
        batch = trainer.evaluate()
        print(batch['rewards'].mean())
