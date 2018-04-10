from typing import Sequence, Optional
import abc
import generative

import torch
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np


class MaskedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, _input, mask):
        masked_weight = self.weight * mask
        return f.linear(_input, masked_weight, self.bias)


class MaskedAutoencoder(generative.ImplicitGenerativeModel, metaclass=abc.ABCMeta):
    def __init__(self, made_network: torch.nn.Module, input_dim: int, hidden_dimensions: Sequence[int],
                 context_dim: int, optimizer: torch.optim.Optimizer, num_masks: int=1, name: str="",
                 *, continuous: bool):
        self._made = made_network

        self._masks = [self._sample_masks(input_dim, hidden_dimensions,
                                          context_dim=context_dim, continuous=continuous)
                       for _ in range(num_masks)]
        self._input_dim = input_dim
        self._optimizer = optimizer
        self._name = name

    @staticmethod
    def _sample_masks(n_in: int, n_hiddens: Sequence[int], continuous: bool, context_dim: int=0):
        # input_ordering = torch.randperm(n_in)
        input_ordering = torch.arange(0, n_in).type(torch.LongTensor)
        if context_dim == 0:
            connectivities = [input_ordering + 1]
        else:
            connectivities = [torch.cat([input_ordering + 1, torch.zeros(context_dim).type(torch.LongTensor)])]
        connectivities.extend([torch.randperm(n_hidden) % n_in for i, n_hidden in enumerate(n_hiddens)])
        if continuous:
            connectivities.append(torch.cat([input_ordering, input_ordering]))
        else:
            connectivities.append(input_ordering)
        masks = [connectivities[i][:, None] >= connectivities[i - 1][None, :] for i in range(1, len(connectivities))]
        masks = [Variable(mask.type(torch.FloatTensor).cuda()) for mask in masks]
        return masks, input_ordering

    def train(self, iteration: int, true_samples: Variable, context: Optional[Variable]=None):
        input_ = self._made(true_samples, self._masks[iteration % len(self._masks)][0], context)
        loss = self._loss(input_, true_samples)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss

    @abc.abstractmethod
    def _loss(self, input_, target):
        pass

    @abc.abstractmethod
    def _sample_dimensions(self, statistics):
        pass

    def generate_samples(self, num_samples: int, context: Optional[Variable]=None) -> Variable:
        mask = np.random.randint(len(self._masks))
        ordering = self._masks[mask][1].numpy()
        samples = Variable(torch.zeros(num_samples, self._input_dim).cuda(), volatile=True)
        for i in range(self._input_dim):
            out = self._sample_dimensions(self._made(samples, self._masks[mask][0], context))
            idx = torch.from_numpy((ordering == i).nonzero()[0].astype(np.int32)).type(torch.cuda.LongTensor)
            samples[:, idx] = out[:, idx]
        return samples

    def evaluate(self, test_data: Variable, context: Optional[Variable]=None):
        mask = np.random.randint(len(self._masks))
        valid_loss = self._loss(self._made(test_data, self._masks[mask][0], context), test_data)
        return valid_loss


class BinaryMaskedAutoencoder(MaskedAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, continuous=False)

    def _loss(self, input_, target):
        return f.binary_cross_entropy_with_logits(input_, target)

    def _sample_dimensions(self, statistics):
        return torch.bernoulli(f.sigmoid(statistics))


class GaussianMaskedAutoencoder(MaskedAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, continuous=True)

    def _loss(self, statistics, target):
        means = statistics[:, :statistics.size(1)//2]
        logstddevs = statistics[:, statistics.size(1)//2:]
        lognorm_constants = 0.5 * np.asscalar(means.size(1)*np.log(2*np.pi).astype(np.float32)) + \
                            logstddevs.sum(dim=1)
        return (0.5 * (((target - means)/torch.exp(logstddevs))**2).sum(dim=1) + lognorm_constants).mean()

    def _sample_dimensions(self, statistics):
        mean = statistics[:, :statistics.size(1)//2]
        logstddev = statistics[:, statistics.size(1)//2:]
        return torch.normal(mean, torch.exp(logstddev))
