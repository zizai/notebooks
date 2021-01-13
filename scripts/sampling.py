import ray
from ray.rllib.utils.memory import ray_get_and_free

from neuroblast import agents as es
from ray.tune.logger import pretty_print

from chaosbreaker.envs.text.writing import Writing


def main():
    ray.init()
    config = es.DEFAULT_CONFIG.copy()
    config['env_config'] = {'root_dir': './data/wiki_test'}
    config['num_workers'] = args.num_workers
    config['episodes_per_batch'] = args.num_episodes
    config['train_batch_size'] = args.num_rollouts
    env = Writing
    trainer = es.ESTrainer.remote(config, env)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = ray_get_and_free(trainer.train.remote())
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = ray_get_and_free(trainer.save.remote())
            print("checkpoint saved at", checkpoint)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_rollouts', type=int, default=10)
    args = parser.parse_args()

    main()
