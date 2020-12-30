#!/usr/bin/python3

import argparse as ap
import decimal
import matplotlib.pyplot as plt
import textwrap

import linear_regression as lr


def predict(storage: str, datafile: str, value: decimal.Decimal):
    model = lr.LinearRegression(storage)
    try:
        model.load()
    except lr.InvalidModel as ex:
        print(f'\033[91m-> Error: {ex}\033[0m')
        return

    prediction = model.predict(value)

    print(f'\033[96m-> -----------------\n'
          f'    value: {prediction.x}\n'
          f'    prediction: {prediction.y}\n\033[0m')
    if prediction.precision is not None and prediction.y_with_precision is not None:
        print(f'\033[96m    RMSE precision: {prediction.y_with_precision}'
              f' ± {prediction.precision}\033[0m')
    else:
        print('\033[93m-> Warning: can`t calculate result with precision, '
              'no precision in storage\033[0m')
        return
    print(f'\033[96m-> -----------------\033[0m')

    if not datafile:
        return

    try:
        data = lr.Data(datafile)
    except lr.InvalidData as ex:
        print(f'\033[91m-> Error: {ex}\n'
              f'Can`t analyze prediction with data\033[0m')
        return

    if not data.point_is_in(prediction.x):
        print(f'\033[93m-> Warning: Can`t analyze prediction, '
              f'it`s out of given data area [{data.x_boarders[0]} .. {data.x_boarders[1]}]\033[0m')
        return

    data.plot()
    model.plot_function(data)
    prediction.plot()
    plt.show()


if __name__ == '__main__':
    class CustomFormatter(ap.ArgumentDefaultsHelpFormatter,
                          ap.RawDescriptionHelpFormatter):
        pass

    parser = ap.ArgumentParser(
        description=textwrap.dedent('''\
        Predict result for [value] based on trained model:

        Y = theta0 + theta1 * X

        Use pre-calculated coefficients theta from [storage] file.
        With given [datafile] show prediction on data's plot.  
        '''),
        formatter_class=CustomFormatter,
    )
    parser.add_argument('-v', metavar='value', type=str, required=True,
                        help='x-value for prediction, for ex. 12.8')
    parser.add_argument('-d', metavar='datafile', type=str,
                        help='file with data for graphics')
    parser.add_argument('-s', metavar='storage', type=str, default='model.json',
                        help='file with calculated coefficients')
    args = parser.parse_args()
    try:
        x_value = decimal.Decimal(args.v)
    except decimal.InvalidOperation:
        print('\033[91m-> Error: Wrong format for value, try again, please\n'
              'for example: 12.8\033[0m')
    else:
        predict(args.s, args.d, x_value)
