## Intrinsic Reward

Variational intrinsic control

VIME: Variational information maximizing exploration

Diversity is all you need

DS-VIC

Infobot

## VAE

AUTOENCODING VARIATIONAL INFERENCE FOR TOPIC MODELS

Dirichlet Variational Autoencoder for Text Modeling

    By the nature of the VAE model, there is a trade-off
    between the NLL, which measures the reconstruction
    quality, and the KL term, which prevent the model 
    from simply memorizing training samples.
    We conduct a experiment on a synthetic data set
    to better understand the behavior of VAE when KL
    divergence is at different levels... 
    
    We can see that when KL term is very small, almost
    all samples get encoded into the same uniform
    Gaussian. When KL term is at a medium
    level, encoded samples are more scattered but still
    have non-zero densities when transitioning from
    the mean value of a sample to another. When KL
    term is very large, the mean values of the encoded
    samples are still not far away from each other in
    the latent space. What is different is that samples
    are encoded into disjoint areas with very small covariances.
    This indicates that the model achieves
    higher reconstruction qualities by trying to memorize
    each sample. Figure 4 indicates that there is
    an optimal level of KL divergence for VAE models. 
    However, how to determine this optimal level of 
    KL divergence remains an open problem.

Nonparametric variational auto-encoders for hierarchical representation learning

Variational Russian Roulette for Deep Bayesian Nonparametrics

Nonparametric variational auto-encoders for hierarchical representation learning

Learning Hierarchical Priors in VAEs

THE DEEP WEIGHT PRIOR


Markov Chain Monte Carlo and Variational Inference: Bridging the Gap

Importance Weighted Hierarchical Variational Inference

*Hierarchical Importance Weighted Autoencoders*

IWAE can be interpretted as using a corrected variational
distribution in the normal variational lower bound.
The proposal is corrected towards the true posterior by the
importance weighting, and approaches the latter with an
increasing number of samples.

Intuitively, when only one sample is drawn to estimate the
variational lower bound, the loss function highly penalizes
the drawn sample, and thus the encoder. The decoder will
be adjusted accordingly to maximize the likelihood in a
biased manner, as it treats the sample as the real, observed
data. In the IWAE setup, the inference model is allowed to
make mistakes, as a sample corresponding to a high loss is
penalized less owing to the importance weight.

Drawing multiple samples also allows us to represent the
distribution at a higher resolution. This motivates us to
construct a joint sampler such that the empirical distribution
drawn from the joint sampler can better represent the
posterior distribution.

*Improving Importance Weighted Auto-Encoders with Annealed Importance Sampling*


The encoder aggregates the element-wise representations of the sequence
with a linear sum. According to GQN paper, this is supposed to work.

    The additive aggregation function was found to work well in practice,
    despite its simplicity. Since the representation and generation
    networks are trained jointly, gradients from the generation network
    encourage the representation network to encode each observation
    independently in such a way that when they are summed element-wise,
    they form a valid scene representation.
