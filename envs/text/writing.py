from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import namedtuple

import ray
import ray.rllib as rllib
from ray.rllib.utils.memory import ray_get_and_free

from neuroblast.datasets.wikipedia import Wikipedia

WritingObservation = namedtuple('WritingObservation', ['current', 'goal', 'messages'])


class Writing(rllib.env.MultiAgentEnv):

    demos = None
    agent = None
    agent_id = None

    def __init__(self, env_config):
        self.root_dir = env_config['root_dir']
        self.read = []
        self.write = []

    @property
    def all_agent_ids(self):
        return ray.objects().keys()

    def _load_demos(self):
        self.demos = Wikipedia(self.root_dir).docs

    def register_agent(self, agent):
        agent_id = ray.put(agent)
        self.agent = agent
        self.agent_id = agent_id
        return agent_id

    def get_remote_agent(self, agent_id):
        return ray.get(agent_id)

    def get_messages(self):
        if self.agent_id is None:
            return
        else:
            return ray_get_and_free(self.agent.pull_messages.remote())

    def send_messages(self, agent_id, msg):
        agent = self.get_remote_agent(agent_id)
        return agent.push_messages.remote(msg)

    def _get_current_and_goal(self, goal_index):
        goal = self.read[goal_index]
        goal_len = len(goal)
        current_len = random.randint(2, goal_len)
        current = goal[:current_len]
        return current, goal

    def step(self, action: str):
        self.write.append(action)
        done = 1 if len(self.read) == len(self.write) else 0
        if done:
            current = None
            goal = None
            messages = None
        else:
            goal_index = len(self.write)
            current, goal = self._get_current_and_goal(goal_index)

            if self.agent_id is None:
                messages = None
            else:
                messages = self.get_messages()

        obs = WritingObservation(current=current, goal=goal, messages=messages)
        # TODO complexity score
        reward = 0
        done = 1 if len(self.read) == len(self.write) else 0
        info = {}
        return obs, reward, done, info

    def reset(self):
        if self.demos is None:
            self._load_demos()

        doc = random.sample(self.demos, 1)[0]

        lines = []
        line2 = ""
        for line in doc['text'][1:]:
            line2 += line
            if len(line2) >= 500:
                lines.append(line2)
                line2 = ""
            elif line == doc['text'][-1]:
                lines.append(line2)
            else:
                continue

        self.read = lines
        self.write = []
        current, goal = self._get_current_and_goal(0)
        messages = self.get_messages()
        obs = WritingObservation(current=current, goal=goal, messages=messages)
        return obs


def test_writing():
    config = {'root_dir': '../data/wiki_test'}
    env = Writing(config)

    class TestAgent(object):
        def __init__(self):
            self.msg_queue = [1]

        def push_messages(self, msg):
            self.msg_queue.append(msg)

        def pull_messages(self):
            return self.msg_queue

    ray.init()
    agent = ray.remote(TestAgent).remote()
    agent_id = env.register_agent(agent)

    test_msg = 'Hi there'
    res = env.send_messages(agent_id, test_msg)
    ray_get_and_free(res)
    assert ray.get(agent.pull_messages.remote()).pop() == test_msg

    obs = env.reset()
    print(obs)
    results = env.step('I want to send a message')
    print(env.write)
    print(results)


if __name__ == '__main__':
    test_writing()
