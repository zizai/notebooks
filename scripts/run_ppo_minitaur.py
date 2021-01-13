import sys
sys.path.append('./')

import pickle
import random
import ray
import time

from ray.tune import register_env

from neuroblast.agents.ppo.ppo import PPOTrainer
from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv
from neuroblast.utils.logger import setup_logger


def main():
    register_env('MinitaurEnv', lambda env_config: MinitaurGymEnv(**env_config))

    if args.eval:
        config = {
            'env_config': {
                'render': True,
            },
            'num_workers': 0,
            'seed': 123,
            'skill_input': random.randint(0, 9),
        }

        trainer = PPOTrainer(config, 'MinitaurEnv')
        state = pickle.load(open(args.restore_checkpoint, "rb"))
        states = pickle.loads(state['worker'])['state']
        trainer.set_weights(states)

        while True:
            trainer.evaluate()
            time.sleep(0.01)
    else:
        log_interval = args.log_interval
        num_episodes = args.num_episodes

        ray.init()

        config = {
            'num_workers': args.num_workers,
            'policy': args.policy,
            'rollout_fragment_length': 200,
            'sgd_minibatch_size': 256,
            'num_sgd_iter': 20,
            'train_batch_size': 3200,
            'use_env_rewards': False,
        }

        trainer = PPOTrainer(config=config, env='MinitaurEnv')

        if args.restore_checkpoint:
            logger.info('Resuming from checkpoint path: {}'.format(args.restore_checkpoint))
            trainer.restore(args.restore_checkpoint)

        for epi_counter in range(num_episodes):
            res = trainer.train()
            logger.info(res['info'])

            if (epi_counter + 1) % log_interval == 0:
                ckp = trainer.save()
                logger.info('model saved to: {}'.format(ckp))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--policy', type=str, default='default')
    parser.add_argument('--restore_checkpoint', type=str)
    args = parser.parse_args()
    logger = setup_logger()
    main()
