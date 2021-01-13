import logging
import pickle
import sys
from pprint import pprint

sys.path.append('./')

import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import ray
from ray.tune import register_env

from neuroblast.agents import ESTrainer
from neuroblast.policy.default_policy import DefaultPolicy
from chaosbreaker.envs.robotics.cheetah import CheetahEnv


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def env_creator(env_config):
    return CheetahEnv(**env_config)


def main():
    if args.address:
        ray.init(address=args.address)
    else:
        ray.init()

    if args.eval is True:
        env = CheetahEnv(render=True)

        config = {
            'model': {
                'sparse': args.sparse_model,
                's_dim': args.state_dim
            }
        }
        pi = DefaultPolicy(env.observation_space, env.action_space, config=config)

        if args.restore_checkpoint:
            state = pickle.load(open(args.restore_checkpoint, "rb"))
            pi.model.set_parameters(state['model_parameters'])
            print(pi.model)

            obs = env.reset()
            for i in range(1000):
                a = pi.compute_single_action(obs)
                obs, rew, _, _ = env.step(a)
                time.sleep(.1)

    else:
        log_interval = 100
        steps = 10000
        time_step = 0.01

        hist1 = []
        hist2 = []
        tt = []

        config = {
            'policy': DefaultPolicy,
            'num_workers': args.num_workers,
            'episodes_per_batch': 1,
            'train_batch_size': 1,
            'env_config': {
                'max_num_steps': 200,
                'render': False,
            },
            'model': {
                'sparse': args.sparse_model is True,
                's_dim': args.state_dim,
            },
        }

        register_env('CheetahEnv', env_creator)
        trainer = ESTrainer(config, 'CheetahEnv')

        if args.restore_checkpoint is not None:
            trainer.restore(args.restore_checkpoint)

        for step_counter in range(steps):
            res = trainer.train()
            logger.info(res['timesteps_total'])

            if (step_counter + 1) % log_interval == 0:
                ckp = trainer.save()
                pprint(ckp)
                pprint(res)

        if args.analysis is True:
            num_motors = 8
            hist1, hist2 = np.array(hist1), np.array(hist2)

            fig, axs = plt.subplots(num_motors, 2, figsize=(12, 16))
            for i in range(num_motors):
                s1, s2 = hist1[:, 2*num_motors + i], hist2[:, 2*num_motors + i]
                axs[i][0].plot(tt, s1, tt, s2)
                axs[i][0].set_xlim(0, 10)
                axs[i][0].set_xlabel('time')
                axs[i][0].set_ylabel('s1 and s2')
                axs[i][0].grid(True)

                cxy, f = axs[i][1].cohere(s1, s2)
                axs[i][1].set_ylabel('coherence')

            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str)
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--restore_checkpoint', type=str)
    parser.add_argument('--sparse_model', action='store_true')
    parser.add_argument('--state_dim', type=int, default=512)
    args = parser.parse_args()

    main()
