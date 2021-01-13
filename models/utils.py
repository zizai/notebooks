import numpy as np
import torch
import torch.nn.functional as F


class LogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))

        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1., range_max+2., 1.).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

            self.log_q = (- (-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[: -n_sample].view(b1, b2, -1)
    sample_w = all_w[- n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[: -n_sample].view(b1, b2)
    sample_b = all_b[- n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = torch.einsum('ijk,ijk->ij', [true_w, inputs]) + true_b - true_log_probs
    samp_logits = torch.einsum('lk,ijk->ijl', [sample_w, inputs]) + sample_b - samp_log_probs
    samp_logits.masked_fill_(hit, -1e30)
    logits = torch.cat([true_logits[:, :, None], samp_logits], -1)

    return logits


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


"""
Implementation of rational-quadratic splines in this file is taken from
https://github.com/bayesiains/nsf.
Thank you to the authors for providing well-documented source code!
"""

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet


def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet
