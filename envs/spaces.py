import gym
import os
import torch

from constants import (MIN_SEQ_LEN, PAD_TOKEN_ID)


class LibraryObservation(gym.spaces.Space):
    """
    A tensor representation over documents

    Parameters
    ----------
    data_dir : string
        directory of dataset

    vocab_path : string
        filepath to vocabulary
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.obs, self.sizes = self._load()
        self.shape = len(self.obs)

    def _load(self):
        docs = self.dataset.docs
        obs = []
        width = len(docs)
        sizes = []

        for i in range(width):
            doc_text = docs[i]['text']

            if not doc_text:
                continue

            line_obs = []
            height = len(doc_text)
            line_sizes = []

            for j in range(height):
                line = doc_text[j]

                if not line:
                    continue

                depth = len(line)
                line_obs.append(line)
                line_sizes.append(depth)

            obs.append(line_obs)
            sizes.append(line_sizes)

        return obs, sizes

    def sample(self):
        n_docs = len(self.sizes)
        i = torch.randint(0, n_docs - 1, (1,)).item()
        n_lines = len(self.sizes[i])
        j = torch.randint(0, n_lines - 1, (1,)).item()
        line = self.lookup_ij(i, j)
        return (i, j), line

    def lookup_i(self, i):
        if i >= len(self.obs):
            return
        else:
            return self.obs[i]

    def lookup_ij(self, i, j):
        if i >= len(self.obs):
            return
        else:
            if j >= len(self.obs[i]):
                return
            else:
                line = self.obs[i][j]
                return line


class LibraryAction(gym.spaces.Space):
    """
    Take a look at some sentences in the library

    Parameters
    ----------
    sizes : list
        lists of lengths of every line of every document
    """

    def __init__(self, sizes):
        self.sizes = sizes
        self.shape = len(self.sizes)

    def sample(self):
        n_obs = len(self.sizes)
        i = torch.randint(0, n_obs - 1, (1,)).item()
        return i
