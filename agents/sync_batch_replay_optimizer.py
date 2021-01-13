import random
from collections import defaultdict

import ray
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.sgd import averaged
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.memory import ray_get_and_free


class SyncBatchReplayOptimizer(PolicyOptimizer):
    """Variant of the sync replay optimizer that replays entire batches.

    This enables RNN support. Does not currently support prioritization."""

    def __init__(self,
                 workers,
                 learning_starts=1000,
                 buffer_size=10000,
                 train_batch_size=1000,
                 num_sgd_iter=1,
                 train_every=100):
        """Initialize a batch replay optimizer.

        Arguments:
            workers (WorkerSet): set of all workers
            learning_starts (int): start learning after this number of
                timesteps have been collected
            buffer_size (int): max timesteps to keep in the replay buffer
            train_batch_size (int): number of timesteps to train on at once
        """
        PolicyOptimizer.__init__(self, workers)

        self.replay_starts = learning_starts
        self.max_buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.num_sgd_iter = num_sgd_iter
        self.train_every = train_every
        assert self.max_buffer_size >= self.replay_starts

        # List of buffered sample batches
        self.replay_buffer = []
        self.buffer_size = 0

        # Stats
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.learner_stats = {}

    @override(PolicyOptimizer)
    def step(self):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.workers.remote_workers():
                batches = ray_get_and_free(
                    [e.sample.remote() for e in self.workers.remote_workers()])
            else:
                batches = [self.workers.local_worker().sample()]

            # Handle everything as if multiagent
            tmp = []
            for batch in batches:
                if isinstance(batch, SampleBatch):
                    batch = MultiAgentBatch({
                        DEFAULT_POLICY_ID: batch
                    }, batch.count)
                tmp.append(batch)
            batches = tmp

            for batch in batches:
                if batch.count > self.max_buffer_size:
                    raise ValueError(
                        "The size of a single sample batch exceeds the replay "
                        "buffer size ({} > {})".format(batch.count,
                                                       self.max_buffer_size))
                self.replay_buffer.append(batch)
                self.num_steps_sampled += batch.count
                self.buffer_size += batch.count
                while self.buffer_size > self.max_buffer_size:
                    evicted = self.replay_buffer.pop(0)
                    self.buffer_size -= evicted.count

        if self.num_steps_sampled >= self.replay_starts and self.num_steps_sampled % self.train_every == 0:
            iter_extra_fetches = defaultdict(list)
            with self.grad_timer:
                for i in range(self.num_sgd_iter):
                    batch_fetches = self._sgd_step()
                    for k, v in batch_fetches.items():
                        iter_extra_fetches[k].append(v)
            self.grad_timer.push_units_processed(self.train_batch_size * self.num_sgd_iter)
            return averaged(iter_extra_fetches)
        else:
            self.grad_timer = TimerStat()
            self.learner_stats = {}
            return {}

    @override(PolicyOptimizer)
    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean, 3),
                "opt_throughput": round(self.grad_timer.mean_throughput, 3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })

    def _sgd_step(self):
        samples = [random.choice(self.replay_buffer)]
        while sum(s.count for s in samples) < self.train_batch_size:
            samples.append(random.choice(self.replay_buffer))
        samples = SampleBatch.concat_samples(samples)
        info_dict = self.workers.local_worker().learn_on_batch(samples)
        for policy_id, info in info_dict.items():
            self.learner_stats[policy_id] = get_learner_stats(info)
        self.num_steps_trained += samples.count
        return info_dict
