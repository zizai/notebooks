import gym

from neuroblast.datasets.wikipedia import Wikipedia

from chaosbreaker.envs.spaces import (LibraryObservation, LibraryAction)


class WikipediaLibrary(gym.Env):
    """
    Parameters
    ----------
    data_dir : string
        data source for observation

    vocab_path : string
        filepath to vocabulary
    """
    def __init__(self, data_dir):
        self.dataset = Wikipedia(data_dir)
        self.observation_space = LibraryObservation(self.dataset)
        self.action_space = LibraryAction(self.observation_space.sizes)

    def step(self, i):
        obs = self.observation_space.lookup_i(i)
        return obs
