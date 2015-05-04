import argparse

from sklearn import linear_model
import pandas as pd
import numpy as np


def split_xy(data, target='GRADE'):
    return data.drop(target, axis=1).values, data[target].values

def split_train_test(data, termnum):
    return data[data.termnum < termnum], data[data.termnum == termnum]

def rmse(predicted, actual):
    """Compute root mean squared error between the predicted values and the
    actual values. This assumes `predicted` and `actual` are arrays of values to
    compare, they are the same length, and there are no missing values.
    """
    sqerror = (predicted - actual) ** 2
    mse = sqerror.sum() / len(sqerror)
    return np.sqrt(mse)

def make_parser():
    parser = argparse.ArgumentParser(
        description='Run decision tree regression/classification on ers data')
    parser.add_argument(
        'data_file', action='store',
        help='data file')
    parser.add_argument(
        '-t', '--type', action='store',
        default='lr', choices=('lr', 'rr'),
        help='lr=linear regression, rr=ridge regression')
    parser.add_argument(
        '-a', '--alpha',
        type=float, default=0.1,
        help='regularization term for ridge regression')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Read data.
    data = pd.read_csv(args.data_file)
    data = data.dropna()
    del data['GRADE']
    target = 'grdpts'

    # Split train/test set.
    train, test = split_train_test(data, 11)

    # Split up predictors/targets.
    train_X, train_y = split_xy(train, target)
    test_X, test_y = split_xy(test, target)

    # Build classifier.
    clf = (linear_model.LinearRegression() if args.type == 'lr'
           else linear_model.Ridge(alpha=args.alpha))
    clf = clf.fit(train_X, train_y)
    print 'Accuracy: %.5f' % clf.score(test_X, test_y)

    # Compute RMSE
    predicted = clf.predict(test_X)
    print 'RMSE:     %.5f' % rmse(predicted, test_y)
