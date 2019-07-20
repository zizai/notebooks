import gym
import torch

from agent import Agent

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('MountainCar-v0').env
    agent = Agent(device, env)
    #obs=env.reset()
    #agent.take_action(obs)
    agent.train(1000, 500)
