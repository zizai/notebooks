import pickle
import sys
import time

sys.path.append("./")

import ray
from neuroblast.agents.a2c import A2CTrainer
from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv
from neuroblast.utils.logger import setup_logger
from ray.tune import register_env


def main():
    register_env("MinitaurEnv", lambda env_config: MinitaurGymEnv(**env_config))

    if args.eval:
        config = {
            "env_config": {
                "render": True,
            },
            "num_workers": 0,
            "seed": 123,
        }

        trainer = A2CTrainer(config, "MinitaurEnv")
        state = pickle.load(open(args.restore_checkpoint, "rb"))
        states = pickle.loads(state["worker"])["state"]
        trainer.set_weights(states)

        while True:
            trainer.workers.local_worker().sample()
            time.sleep(0.01)
    else:
        ray.init()

        config = {
            "num_workers": args.num_workers,
            "rollout_fragment_length": 50,
            "train_batch_size": 2500,
            "num_sgd_iter": 80
        }
        trainer = A2CTrainer(config, "MinitaurEnv")

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
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--num_episodes", type=int, default=50000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--policy", type=str, default="default")
    parser.add_argument("--restore_checkpoint", type=str)
    args = parser.parse_args()
    logger = setup_logger()
    main()
