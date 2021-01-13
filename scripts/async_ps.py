from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import time

import ray

parser = argparse.ArgumentParser(description="Run the asynchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=4, type=int,
                    help="The number of workers to use.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")


@ray.remote
class ParameterServer(object):
    def __init__(self):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [random.random() for i in range(5)]
        keys = [i for i in range(len(values))]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value

    def pull(self, keys):
        return [self.weights[key] for key in keys]


@ray.remote
def worker_task(ps, worker_index, batch_size=50):
    return


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(redis_address=args.redis_address)

    ps = ParameterServer.remote()
    ps1 = ParameterServer.remote()

    object_ids = [ray.put(ps), ray.put(ps1)]

    print(ray.objects())
    print([ray.get(o.pull.remote([0, 1])) for o in ray.get(object_ids)])
