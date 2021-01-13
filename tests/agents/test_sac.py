import ray
from ray.tune import register_env

from chaosbreaker.envs.robotics.minitaur import MinitaurGymEnv
from neuroblast.agents.sac.sac import SACTrainer


def test_sac():
    ray.init()

    config = {
        "model": {
            "use_lstm": False,
        },
        "use_state_preprocessor": False,
        "learning_starts": 200,
        "normalize_actions": True,
        "rollout_fragment_length": 10,
        "timesteps_per_iteration": 100,
        "num_gpus": 1,
        "num_workers": 0,
        "evaluation_interval": 0,
        "monitor": False,
    }
    register_env("MinitaurEnv", lambda env_config: MinitaurGymEnv(**env_config))
    trainer = SACTrainer(config, "MinitaurEnv")

    print(trainer.get_policy().model)

    for i in range(5):
        res = trainer.train()
        print(res)

    print(trainer._logdir)

    ray.shutdown()
