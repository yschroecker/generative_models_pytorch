from typing import Optional
import abc

from torch.autograd import Variable


class ImplicitGenerativeModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_samples(self, batch_size: int, context: Optional[Variable]) -> Variable:
        pass

    @abc.abstractmethod
    def train(self, iteration: int, true_samples: Variable, context: Optional[Variable]):
        pass

