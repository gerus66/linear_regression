from typing import List, NamedTuple, Optional

import csv
from decimal import Decimal, InvalidOperation
import json
import math
import matplotlib.pyplot as plt
import os.path


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


class InvalidModel(Exception):
    pass


class LinearRegression:
    theta0: Decimal = Decimal('0')
    theta1: Decimal = Decimal('0')
    precision: Optional[Decimal] = None
    learning_rate: Decimal
    storage: str
    errors: List[Decimal]

    def __init__(self, storage: str, learning_rate: Decimal = Decimal('0.01')):
        """
        :param storage: file for coefficients
        :param learning_rate: coefficient for training
        """
        if not Decimal('0') < learning_rate < Decimal('1'):
            raise InvalidModel('Learning rate should be in range (0, 1)')
        self.learning_rate = learning_rate
        self.storage = storage
        if not os.path.isfile(self.storage):
            self._reset_file()
        self.errors = []

    def load(self):
        coeffs = dict()
        with open(self.storage, 'r') as storage:
            line = storage.readline()
        try:
            coeffs = json.loads(line)
        except json.decoder.JSONDecodeError:
            self._reset_file()
        if 'theta0' not in coeffs or 'theta1' not in coeffs:
            raise InvalidModel('Coefficients aren`t set, you should train your model')

        try:
            self.theta0 = Decimal(coeffs['theta0'])
            self.theta1 = Decimal(coeffs['theta1'])
            if 'precision' in coeffs:
                self.precision = Decimal(coeffs['precision'])
        except InvalidOperation:
            raise InvalidData('Coefficients aren`t numbers')

    def predict(self, x_value: Decimal) -> Prediction:
        """
        :param x_value: request for prediction
        :return: predicted result by precision of request value;
                precision
        """
        x_prec = -x_value.as_tuple().exponent
        row_result = self._linear(x_value)
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
            [self._linear(data.x_boarders[0]), self._linear(data.x_boarders[1])],
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
            self._one_round(xs, data.points_y)
            self.errors.append(self._loss_function(xs, data.points_y))

        self.theta1 *= data.x_norm_coeff
        self.theta0 -= self.theta1 * data.x_boarders[0]
        if not self.errors:
            raise InvalidModel('Too few training iterations')
        self.precision = self.errors[-1]
        self._dump()

    def _linear(self, x: Decimal) -> Decimal:
        """
        :param x: x value
        :return: row prediction result (without precision)
        """
        return self.theta0 + self.theta1 * x

    def _get_round_factor(self) -> Optional[int]:
        if not self.precision:
            return None
        prec = self.precision.as_tuple()
        return -len(prec.digits) - prec.exponent + 1

    def _dump(self):
        f = open(self.storage, 'w')
        coefficients = {
            'theta0': str(self.theta0),
            'theta1': str(self.theta1),
            'precision': str(self.precision),
        }
        f.write(json.dumps(coefficients))

    def _reset_file(self):
        with open(self.storage, 'w') as storage:
            storage.write('{}')
        print(f'-> Reset file "{self.storage}"')

    def _one_round(self, xs: List[Decimal], ys: List[Decimal]):
        """
        :param xs: array of X for training
        :param ys: array of Y for training
        """
        x_error = sum(self._linear(x) - y for x, y in zip(xs, ys)) / len(xs)
        theta0_tmp = self.theta0 - self.learning_rate * x_error

        y_error = sum((self._linear(x) - y) * x for x, y in zip(xs, ys)) / len(xs)
        theta1_tmp = self.theta1 - self.learning_rate * y_error

        self.theta0 = theta0_tmp
        self.theta1 = theta1_tmp

    def _loss_function(self, xs: List[Decimal], ys: List[Decimal]) -> Decimal:
        """
        :param xs: array of X for training
        :param ys: array of Y for training
        :return: Root Mean Squared Error for calculated coefficients
        """
        errors = [(self._linear(x) - y)**2 for x, y in zip(xs, ys)]
        return Decimal(math.sqrt(sum(errors) / len(xs)))
