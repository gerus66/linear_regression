from typing import Callable, List, NamedTuple, Optional
import csv
from decimal import Decimal, InvalidOperation
import json
import math
import matplotlib.pyplot as plt
import os.path

from . import regression as base_reg


class InvalidData(Exception):
    pass


class Data:
    points_x: List[Decimal]
    points_y: List[Decimal]
    x_alias: str
    y_alias: str
    x_boarders: (Decimal, Decimal)  # x_min, x_max
    x_norm_coeff: Decimal

    def __init__(self, datafile: str):
        """
        :param datafile: csv file with data for training your model,
        should has exactly 2 columns
        """
        if not os.path.isfile(datafile):
            raise InvalidData(f'No such datafile {datafile}')

        self.points_x = []
        self.points_y = []
        with open(datafile, 'r') as table:
            reader = csv.DictReader(table, delimiter=',')
            if not reader.fieldnames or len(reader.fieldnames) != 2:
                raise InvalidData('Can`t load data, there should be exactly 2 columns')

            self.x_alias, self.y_alias = reader.fieldnames
            try:
                for row in reader:
                    self.points_x.append(Decimal(row[self.x_alias]))
                    self.points_y.append(Decimal(row[self.y_alias]))
            except InvalidOperation:
                raise InvalidData('Can`t load data, there should be numbers')

        if not self.points_x:
            raise InvalidData('No data in file')

        self.x_boarders = min(self.points_x), max(self.points_x)
        if self.x_boarders[0] == self.x_boarders[1]:
            raise InvalidData('All x-values in file are equal, can`t train model on it')

        self.x_norm_coeff = 1 / (self.x_boarders[1] - self.x_boarders[0])

    def normalize_xs(self) -> List[Decimal]:
        """
        :return: list of normalized values in [0, 1] segment
        """
        return [(x - self.x_boarders[0]) * self.x_norm_coeff for x in self.points_x]

    def point_is_in(self, point_x: Decimal) -> bool:
        """
        :param point_x: x-value
        :return: True if x-value is in range of data x-values
        """
        return self.x_boarders[0] <= point_x <= self.x_boarders[1]

    def plot(self):
        plt.title('Training data')
        plt.xlabel(self.x_alias)
        plt.ylabel(self.y_alias)
        plt.plot(self.points_x, self.points_y, 'mo')


class Prediction(NamedTuple):
    x: Decimal
    y: Decimal
    y_with_precision: Optional[Decimal] = None
    precision: Optional[Decimal] = None

    def plot(self):
        plt.title('Prediction with RMS error')
        plt.plot(self.x, self.y_with_precision, 'go')
        plt.errorbar(
            self.x, self.y_with_precision, self.precision,
            lw=2, capsize=5, capthick=2, color='g',
        )


class Linear(base_reg.Regression):
    def loss(self, x: Decimal, y: Decimal) -> Decimal:
        """
        loss function
        :param x: input
        :param y: output prediction
        :return: error for self._func(x) on given prediction
        """
        return (self.func(x) - y)**2 / 2

    @property
    def _regressors(self) -> List[Callable[[Decimal], Decimal]]:
        """
        :return: list of regressor's functions
        """
        return [lambda x: 1, lambda x: x]

    def _gradient(self, x: Decimal, y: Decimal) -> List[Decimal]:
        """
        gradient vector for self._loss() in (x,y)
        """
        return [(self.func(x) - y) * reg(x) for reg in self._regressors]

    def rescale_thetas(self, norm_coef: Decimal, left_x: Decimal):
        self._thetas[1] = self._thetas[1] * norm_coef
        self._thetas[0] = self._thetas[0] - self._thetas[1] * left_x


class InvalidModel(Exception):
    pass


class Manager:
    model: Linear
    precision: Optional[Decimal] = None
    storage: str
    errors: List[Decimal]

    def __init__(self, storage: str, learning_rate: Optional[Decimal] = None):
        """
        :param storage: file for coefficients
        :param learning_rate: coefficient for training
        """
        try:
            self.model = Linear(learning_rate)
        except base_reg.InvalidRegression as ex:
            raise InvalidModel(str(ex))
        self.storage = storage
        if not os.path.isfile(self.storage):
            self._reset_file()
        self.errors = []

    def load(self):
        """
        load coefficients from file
        :return:
        """
        coeffs = dict()
        with open(self.storage, 'r') as storage:
            line = storage.readline()
        try:
            coeffs = json.loads(line)
        except json.decoder.JSONDecodeError:
            self._reset_file()
        if 'coefficients' not in coeffs:
            raise InvalidModel(
                'Coefficients aren`t set, you should train your model'
            )

        try:
            self.model.set_thetas([Decimal(x) for x in coeffs['coefficients']])
            if 'precision' in coeffs:
                self.precision = Decimal(coeffs['precision'])
        except (InvalidOperation, base_reg.InvalidRegression):
            raise InvalidData('Coefficients aren`t valid')

    def predict(self, x_value: Decimal) -> Prediction:
        """
        :param x_value: request for prediction
        :return: predicted result by precision of request value;
                precision
        """
        x_prec = -x_value.as_tuple().exponent
        row_result = self.model.func(x_value)
        y_prec = self._get_round_factor()
        return Prediction(
            x=x_value,
            y=round(row_result, x_prec),
            y_with_precision=round(row_result, y_prec) if y_prec else None,
            precision=round(self.precision, y_prec) if y_prec else None,
        )

    def plot_errors(self):
        plt.title('RMS errors')
        plt.xlabel('training iterations')
        plt.ylabel('log(errors)')
        plt.plot(range(len(self.errors)),
                 [math.log(er) for er in self.errors], color='r')

    def plot_function(self, data: Data):
        """
        :param data: plot function by trained coefficients
        """
        plt.xlabel(data.x_alias)
        plt.ylabel(data.y_alias)
        plt.plot(
            data.x_boarders,
            [
                self.model.func(data.x_boarders[0]),
                self.model.func(data.x_boarders[1]),
            ],
            color='k',
        )

    def train(self, data: Data, iterations: int):
        """
        Dump calculated coefficients to storage
        :param data: preloaded Data for training
        :param iterations: number of training iterations
        """
        xs = data.normalize_xs()
        for i in range(iterations):
            self.model.one_round(xs, data.points_y)
            errors = [
                self.model.loss(x, y) for x, y in zip(xs, data.points_y)
            ]
            self.errors.append(Decimal(math.sqrt(sum(errors) / len(xs))))

        self.model.rescale_thetas(data.x_norm_coeff, data.x_boarders[0])
        if not self.errors:
            raise InvalidModel('Too few training iterations')
        self.precision = self.errors[-1]
        self._dump()

    def _get_round_factor(self) -> Optional[int]:
        if not self.precision:
            return None
        prec = self.precision.as_tuple()
        return -len(prec.digits) - prec.exponent + 1

    def _dump(self):
        f = open(self.storage, 'w')
        coefficients = {
            'coefficients': [str(x) for x in self.model.thetas],
            'precision': str(self.precision),
        }
        f.write(json.dumps(coefficients))

    def _reset_file(self):
        with open(self.storage, 'w') as storage:
            storage.write('{}')
        print(f'-> Reset file "{self.storage}"')
