import numpy as np
import torch
from agents.ppo.ppo_neuroblast_policy import PPONeuroblastPolicy
from ray.rllib.agents import Trainer, with_common_config
from ray.rllib.optimizers import SyncSamplesOptimizer
from agents.ppo.ppo_torch_policy import PPOTorchPolicy

DEFAULT_CONFIG = with_common_config({
    'alpha': 0.1,
    'clip_ratio': 0.2,
    'gamma': 0.99,
    'lambda': 0.97,
    'lr_pi': 3e-4,
    'lr_vf': 1e-3,
    'max_episode_len': 1000,
    'model_hidden_sizes': (256, 128, 64),
    'policy': 'default',
    'num_workers': 4,
    'num_sgd_iter': 80,
    'num_skills': 10,
    'rollout_fragment_length': 200,
    'seed': 123,
    'sgd_minibatch_size': 128,
    'skill_input': None,
    'target_kl': 0.01,
    'train_batch_size': 4000,
    'use_diayn': True,
    'use_env_rewards': True,
    'use_gae': True,
})

policy_options = {
    'default': PPOTorchPolicy,
    'neuroblast': PPONeuroblastPolicy,
}


class PPOTrainer(Trainer):

    _name = "PPO"
    _default_config = DEFAULT_CONFIG

    def _init(self, config, env_creator):
        # Random seed
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env_config = config['env_config']
        self.num_sgd_iter = config['num_sgd_iter']
        self.num_workers = config['num_workers']
        self.sgd_minibatch_size = config['sgd_minibatch_size']
        self.train_batch_size = config['train_batch_size']

        # Set up workers
        policy_cls = policy_options[config['policy']]
        self.workers = self._make_workers(env_creator, policy_cls, config, self.num_workers)
        self.optimizer = SyncSamplesOptimizer(self.workers,
                                              num_sgd_iter=self.num_sgd_iter,
                                              train_batch_size=self.train_batch_size,
                                              sgd_minibatch_size=self.sgd_minibatch_size)

    def _train(self):
        self.optimizer.step()

        res = dict(
            timesteps_this_iter=self.optimizer.num_steps_sampled,
            info=self.optimizer.stats()
        )
        return res

    def evaluate(self):
        return self.workers.local_worker().sample()
