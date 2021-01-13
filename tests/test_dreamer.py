from pprint import pprint

import ray

from chaosbreaker.env_creator import register_envs
from neuroblast.agents.dreamer.dreamer import DreamerTrainer


def test_dreamer():
    ray.init()

    config = {
        "learning_starts": 200,
        "normalize_actions": True,
        "batch_size": 10,
        "rollout_fragment_length": 30,
        "timesteps_per_iteration": 3,
        "num_gpus": 1,
        "num_workers": 0,
    }
    register_envs()
    trainer = DreamerTrainer(config, "MinitaurPixEnv")

    print(trainer.get_policy().model)

    for i in range(5):
        res = trainer.train()
        pprint(res["info"])

    ray.shutdown()
