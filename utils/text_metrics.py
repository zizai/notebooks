from functools import reduce
import sys
import os.path as osp
from random import random

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from scipy.stats import gaussian_kde


def compute_bleu(pred, data, pad_idx):
    """Computes the weighted corpus BLEU of predicted sentences.

    Args:
        pred(list): [num_sentences, max_len]. Predictions in index form.
        data(list): [num_sentences, max_len]. Gold standard indices.

    Return:
        float: corpus weighted BLEU for 1- to 4-grams between 0 and 100.
    """
    pred = [remove_padding(p, pad_idx) for p in pred]
    data = [[remove_padding(d, pad_idx)] for d in data]
    return corpus_bleu(data, pred) * 100


def compute_novelty(sentences, corpus_file, opt, idx_to_word):
    """Computes the novelty of a batch of sentences given a corpus."""
    # Prepare sampled sentences and corpus to compare to
    ref = sentences[0].split("\n")
    sentences = [s.split(" ") for s in sentences[1].split("\n")]
    with open(corpus_file, 'r') as f:
        corpus = [s.rstrip().split(" ") for s in f.readlines()]

    # Remove sentences much longer than the sampled sentences length
    corpus = [s for s in corpus if len(s) < opt.sample_len + 5]

    # Compute the novelty for each sentence
    novelty = []
    closest = []
    for i, sen in enumerate(sentences):
        print("Computing novelty for sentence {}/{}.\n".format(i, len(sentences)))
        mindex = np.argmin(np.array([ter(sen, s) for s in corpus]))
        novelty.append(ter(sen, corpus[mindex]))
        closest.append(" ".join([idx_to_word[int(idx)] for idx in corpus[mindex]]))
        print("Novelty: {}, Sentence: {}, Closest: {}\n".format(novelty[i], ref[i], closest[i]))
    return sum(novelty) / float(len(novelty)), sorted(zip(novelty, ref, closest))


def remove_padding(sentence, pad_idx):
    """Removes the paddings from a sentence"""
    try:
        return sentence[:sentence.index(pad_idx)]
    except ValueError:
        return sentence


def compute_active_units(mu, delta):
    """Computes an estimate of the number of active units in the latent space.

    Args:
        mu(torch.FloatTensor): [n_samples, z_dim]. Batch of posterior means.
        delta(float): variance threshold. Latent dimensions with a variance above this threshold are active.

    Returns:
        int: the number of active dimensions.
    """
    outer_expectation = torch.mean(mu, 0)**2
    inner_expectation = torch.mean(mu**2, 0)
    return torch.sum(inner_expectation - outer_expectation > delta).item()


def compute_accuracy(pred, data):
    """Computes the accuracy of predicted sequences.

    Args:
        pred(torch.Tensor): predictions produces by a model in index form, can be both a vector of last words
            and a matrix containing a batch of predicted sequences.
        data(torch.Tensor): the gold standard to compare the predictions against.

    Returns:
        float: the fraction of correct predictions.
    """
    # Handle different cases, 1 vs all outputs
    denom = reduce(lambda x, y: x * y, pred.shape)

    if len(data) == 1:
        target = data[0][:, 1:]
    else:
        # Here we have to ignore padding from the computation
        target = data[0][:, 1:]
        pred[data[2][:, 1:] == 0] = -1
        denom = denom - torch.sum(1. - data[2])

    return float(torch.eq(target, pred).sum()) / denom


def compute_perplexity(log_likelihoods, seq_lens):
    """Computes a MC estimate of perplexity per word based on given likelihoods/ELBO.

    Args:
        log_likelihoods(list of float): likelihood or ELBO from N runs over the same data.
        seq_lens(list of int): the length of sequences in the data, for computing an average.

    Returns:
        perplexity(float): perplexity per word of the data.
        variance(float): variance of the log_likelihoods/ELBO that were used to compute the estimate.
    """
    # Compute perplexity per word and variance of perplexities in the samples
    perplexity = np.exp(np.array(log_likelihoods).mean() / np.array(seq_lens).mean())
    if len(log_likelihoods) > 1:
        variance = np.array(log_likelihoods).mean(axis=1).std(ddof=1)
    else:
        variance = 0.0

    return perplexity, variance
