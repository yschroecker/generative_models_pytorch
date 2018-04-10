from typing import Optional

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as f

import generative
import visualization


class VAE(generative.ImplicitGenerativeModel):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 latent_size: int, cuda: bool, infovae: bool=False):
        self._encoder = encoder
        self._decoder = decoder
        self._optimizer = optimizer
        self._cuda = cuda
        self._latent_size = latent_size
        self._infovae = infovae

    def _sample(self, means: Variable, logvar: Variable) -> Variable:
        noise = Variable(means.data.new(means.size()).normal_())
        return noise * (logvar * 0.5).exp() + means

    def _kl_divergence(self, encoded_means, encoded_logvar):
        return -0.5 * torch.sum(1 + encoded_logvar - encoded_means**2 - encoded_logvar.exp())

    def _mmd(self, encodings):
        def kernel(x: Variable, y: Variable):
            x = x[:, None, :]
            y = y[None, :, :]
            return (-((x - y)**2).mean(dim=2)/x.size(2)).exp().sum()
        mmd_samples = 200
        sample = Variable(torch.randn((mmd_samples, self._latent_size)))
        if self._cuda:
            sample = sample.cuda()
        return kernel(encodings, encodings) + kernel(sample, sample) - 2 * kernel(encodings, sample)

    def generate_samples(self, batch_size: int, context: Optional[Variable]=None) -> Variable:
        latent_variables = Variable(torch.randn((batch_size, self._latent_size)))
        if self._cuda:
            latent_variables = latent_variables.cuda()
        if context is not None:
            return self._decoder(latent_variables, context)
        else:
            return self._decoder(latent_variables)

    def train(self, iteration: int, true_samples: Variable, context: Optional[Variable]=None):
        if context is None:
            encoded_means, encoded_logvar = self._encoder(true_samples)
            encoded_samples = self._sample(encoded_means, encoded_logvar)
            reconstructed_sample = self._decoder(encoded_samples)
        else:
            encoded_means, encoded_logvar = self._encoder(true_samples, context)
            encoded_samples = self._sample(encoded_means, encoded_logvar)
            reconstructed_sample = self._decoder(encoded_samples, context)
        recostruction_loss = f.mse_loss(reconstructed_sample, true_samples, size_average=False)
        elbo = recostruction_loss
        if self._infovae:
            mmd = self._mmd(encoded_samples)
            elbo += mmd
            visualization.global_summary_writer.add_scalar("VAE: mmd", np.array(mmd[0]), iteration)
        else:
            kldiv = self._kl_divergence(encoded_means, encoded_logvar)
            elbo += kldiv
            visualization.global_summary_writer.add_scalar("VAE: kl", np.array(kldiv[0]), iteration)
        if iteration % 100 == 0:
            visualization.global_summary_writer.add_scalar("VAE: ELBO", np.array(elbo[0]), iteration)
            visualization.global_summary_writer.add_scalar("VAE: reconstruction", np.array(recostruction_loss[0]), iteration)
            visualization.global_summary_writer.add_scalar("VAE: logvar", encoded_logvar.mean(), iteration)
        self._optimizer.zero_grad()
        elbo.backward()
        self._optimizer.step()
