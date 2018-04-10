from typing import Optional

import torch
import torch.utils.data
import numpy as np
import visualization
import generative
from torch.autograd import Variable


class WGAN(generative.ImplicitGenerativeModel):
    def __init__(self, generator: torch.nn.Module, critic: torch.nn.Module, cuda: bool,
                 latent_distribution: torch.distributions.Distribution, generator_optimizer: torch.optim.Optimizer,
                 critic_optimizer: torch.optim.Optimizer, generator_frequency: int, critic_frequency: int=1,
                 gradient_penalty: float=10, critic_clip: float=0, use_loss_guidance: float=0, name: str=''):
        self.__generator = generator
        self.__critic = critic
        self._latent_distribution = latent_distribution
        self._generator_optimizer = generator_optimizer
        self._critic_optimizer = critic_optimizer
        self._gradient_penalty_factor = gradient_penalty
        self._critic_clip = critic_clip
        self._cuda = cuda
        self._generator_frequency = generator_frequency
        self._critic_frequency = critic_frequency
        self._loss_guidance = use_loss_guidance
        self._name = name

    def _generator(self, latent_variables: Variable, context: Optional[Variable]) -> \
            Variable:
        if context is None:
            return self.__generator(latent_variables)
        else:
            return self.__generator(latent_variables, context)

    def _critic(self, samples: Variable, context: Optional[Variable]) -> \
            Variable:
        if context is None:
            return self.__critic(samples)
        else:
            return self.__critic(samples, context)

    def generate_samples(self, n: int, context: Optional[Variable]=None) -> Variable:
        latent_variables = Variable(self._latent_distribution.sample_n(n), requires_grad=False)
        if self._cuda:
            latent_variables = latent_variables.cuda()
        generator_samples = self._generator(latent_variables, context)
        return generator_samples

    def step_critic(self, iteration: int, true_samples: Variable,
                    context: Optional[Variable]=None,
                    weights: Optional[Variable]=None):
        if weights is None:
            weights_tensor = torch.ones(true_samples.size(0))
            if self._cuda:
                weights_tensor = weights_tensor.cuda()
            weights = Variable(weights_tensor)
        generator_samples = self.generate_samples(true_samples.size(0), context)
        critic_loss = (weights * (self._critic(generator_samples, context) - self._critic(true_samples, context))) \
            .mean()
        if iteration % 100 == 0:
            samples = generator_samples.data
            if self._cuda:
                samples = samples.cpu()
            samples = samples.numpy()
            visualization.global_summary_writer.add_scalar(f"WGAN{self._name}: variation", np.array(np.std(samples)),
                                                           iteration)
            visualization.global_summary_writer.add_scalar(f"WGAN{self._name}: true variation", np.array(np.std(true_samples.data.cpu().numpy())),
                                                           iteration)
        if self._loss_guidance > 0:
            assert context is not None
            other_indices = np.arange(context.size(0))
            np.random.shuffle(other_indices)
            other_indices = torch.autograd.Variable(torch.from_numpy(other_indices))
            if self._cuda:
                other_indices = other_indices.cuda()
            other_context = context[other_indices]
            critic_loss -= (weights * self._loss_guidance * (self._critic(true_samples, other_context) - self._critic(true_samples, context))).mean()
        if self._gradient_penalty_factor > 0:
            combination_weights = torch.rand((true_samples.size(0),) + (1,)*(len(true_samples.size()) - 1))
            if self._cuda:
                combination_weights = combination_weights.cuda()
            combination_weights = Variable(combination_weights, requires_grad=False)
            convex_combination = combination_weights * generator_samples + (1 - combination_weights) * true_samples
            convex_combination = Variable(convex_combination.data, requires_grad=True)
            critic_out = self._critic(convex_combination, context)
            gradient = torch.ones(critic_out.size())
            if self._cuda:
                gradient = gradient.cuda()
            gradient = torch.autograd.grad(critic_out, convex_combination, create_graph=True,
                                           grad_outputs=[gradient])
            gradient_penalty = (weights * (gradient[0].norm(2, dim=1) - 1)**2).mean()
            visualization.global_summary_writer.add_scalar(f"WGAN{self._name}: critic loss (no penalty)", critic_loss, iteration)
            critic_loss += self._gradient_penalty_factor * gradient_penalty

        self._critic_optimizer.zero_grad()
        visualization.global_summary_writer.add_scalar(f"WGAN{self._name}: critic loss", critic_loss, iteration)
        critic_loss.backward()
        self._critic_optimizer.step()
        if self._critic_clip > 0:
            for parameter in self.__critic.parameters():
                parameter.data.clamp_(-self._critic_clip, self._critic_clip)

    def step_generator(self, iteration: int, batch_size: int, context: Optional[Variable]=None):
        samples = self.generate_samples(batch_size, context)
        loss = -self._critic(samples, context).mean()

        visualization.global_summary_writer.add_scalar(f"WGAN{self._name}: generator loss", loss, iteration)
        self._generator_optimizer.zero_grad()
        loss.backward()
        self._generator_optimizer.step()

    def train(self, iteration: int, true_samples: Variable, context: Optional[Variable]):
        if iteration % self._critic_frequency == 0:
            self.step_critic(iteration, true_samples, context)
        if iteration % self._generator_frequency == 0:
            self.step_generator(iteration, true_samples.size(0), context)
