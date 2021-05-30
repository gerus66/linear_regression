import abc
from typing import Callable, List, Optional

from decimal import Decimal


class InvalidRegression(Exception):
    pass


class Regression:
    _thetas: List[Decimal]
    _learning_rate: Decimal = Decimal('0.01')

    def __init__(self, learning_rate: Optional[Decimal] = None):
        """
        :param learning_rate: how fast gradient descent will be performed
        """
        self._thetas = [Decimal('0')] * len(self._regressors)
        if learning_rate is not None:
            if not Decimal('0') < learning_rate < Decimal('1'):
                raise InvalidRegression(
                    'Learning rate should be in range (0, 1)'
                )
            self._learning_rate = learning_rate

    @property
    def thetas(self) -> List[Decimal]:
        return self._thetas

    def set_thetas(self, thetas: List[Decimal]):
        if len(thetas) != len(self._regressors):
            raise InvalidRegression('assent to set invalid thetas')
        self._thetas = thetas

    def func(self, x: Decimal) -> Decimal:
        """
        function to fit f(x) (without precision)
        """
        return sum(
            coef * reg(x) for coef, reg in zip(self._thetas, self._regressors)
        )

    @abc.abstractmethod
    def loss(self, x: Decimal, y: Decimal) -> Decimal:
        """
        loss function
        :param x: input
        :param y: output prediction
        :return: error for self._func(x) on given prediction
        """

    def one_round(self, xs: List[Decimal], ys: List[Decimal]):
        """
        Run one round of gradient descent, reset self._thetas
        :param xs: list of X for training
        :param ys: list of Y for training
        """
        assert len(xs) == len(ys), 'invalid data'
        gradients = [self._gradient(x, y) for x, y in zip(xs, ys)]
        average = [sum(items) / len(xs) for items in zip(*gradients)]
        self._thetas = [
            old - self._learning_rate * grad
            for old, grad in zip(self._thetas, average)
        ]

    @property
    @abc.abstractmethod
    def _regressors(self) -> List[Callable[[Decimal], Decimal]]:
        """
        :return: list of regressor's functions
        """

    @abc.abstractmethod
    def _gradient(self, x: Decimal, y: Decimal) -> List[Decimal]:
        """
        gradient vector for self._loss() in (x,y)
        """
