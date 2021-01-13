import pickle
import sys
import time
sys.path.append("./")

import ray

from chaosbreaker.env_creator import register_envs
from neuroblast.agents.sac.sac import SACTrainer
from neuroblast.utils.logger import setup_logger


def main():
    register_envs()
    ray.init()

    if args.eval:
        config = {
            "env_config": {
                "render": True,
            },
            "num_workers": 0,
            "seed": 123,
        }

        trainer = SACTrainer(config, args.env)
        trainer.restore(args.restore_checkpoint)

        while True:
            trainer.workers.local_worker().sample()
            time.sleep(0.01)
    else:
        config = {
            "env_config": {
                "max_num_steps": 1000,
            },
            "initial_alpha": args.initial_alpha,
            "num_gpus": 1,
            "num_workers": args.num_workers,
            "reward_scale": args.reward_scale,
            "rollout_fragment_length": 1,
            "timesteps_per_iteration": 500,
            "seed": 123,
        }
        trainer = SACTrainer(config, args.env)

        if args.restore_checkpoint:
            logger.info("Resuming from checkpoint path: {}".format(args.restore_checkpoint))
            trainer.restore(args.restore_checkpoint)

        for epi_counter in range(args.num_episodes):
            res = trainer.train()
            logger.info(res["info"])

            if (epi_counter + 1) % args.log_interval == 0:
                ckp = trainer.save()
                logger.info("model saved to: {}".format(ckp))

        ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MinitaurEnv")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--initial_alpha", type=float, default=0.1)
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--policy", type=str, default="default")
    parser.add_argument("--restore_checkpoint", type=str)
    args = parser.parse_args()
    logger = setup_logger()
    main()
