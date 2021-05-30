#!/usr/bin/python3

import argparse as ap
import decimal
import matplotlib.pyplot as plt
import textwrap

from src import linear_regression as lr


def training(
        datafile: str, storage: str, iter: int, learning_rate: decimal.Decimal,
):
    try:
        data = lr.Data(datafile)
    except lr.InvalidData as ex:
        print(f'\033[91m-> Error: {ex}\033[0m')
        return

    try:
        model = lr.Manager(storage, learning_rate)
    except lr.InvalidModel as ex:
        print(f'\033[91m-> Error: {ex}\033[0m')
        return

    model.train(data, iter)
    print(f'\033[96m-> Coefficients are dumped to {model.storage}\033[0m')

    plt.subplot(311)
    data.plot()
    model.plot_function(data)
    plt.subplot(313)
    model.plot_errors()
    plt.show()


if __name__ == '__main__':
    class CustomFormatter(ap.ArgumentDefaultsHelpFormatter,
                          ap.RawDescriptionHelpFormatter):
        pass

    parser = ap.ArgumentParser(
        description=textwrap.dedent('''\
        Train model on data from [datafile] using linear regression:
        
        Y = theta0 + theta1 * X
        
        Save calculated coefficients theta and precision
        to [storage] file in JSON string format.
        '''),
        formatter_class=CustomFormatter,
    )
    parser.add_argument('-d', metavar='datafile', type=str, default='data/data.csv',
                        help='file with data for training')
    parser.add_argument('-i', metavar='iterations', type=int, default=10000,
                        help='number of training iterations')
    parser.add_argument('-lr', metavar='rate', type=str, default='0.01',
                        help='learning rate')
    parser.add_argument('-s', metavar='storage', type=str, default='model.json',
                        help='file to store calculated coefficients')
    args = parser.parse_args()
    try:
        learning_rate = decimal.Decimal(args.lr)
    except decimal.InvalidOperation:
        print('\033[91m-> Error: Wrong format for learning rate, try again, please\n'
              'for example: 0.01\033[0m')
    else:
        training(args.d, args.s, args.i, learning_rate)
