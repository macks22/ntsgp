import argparse

from sklearn import tree
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
        '-t', '--task', action='store', default='r',
        help='r=regression, c=classification')
    parser.add_argument(
        '-m', '--max-depth',
        type=int, default=4,
        help='max depth to grow the tree to')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Read data.
    data = pd.read_csv(args.data_file)
    data = data.dropna()
    if args.task == 'r':
        DecisionTree = tree.DecisionTreeRegressor
        del data['GRADE']
        target = 'grdpts'
    else:
        DecisionTree = tree.DecisionTreeClassifier
        del data['grdpts']
        target = 'GRADE'

    # Split train/test set.
    train, test = split_train_test(data, 11)

    # Split up predictors/targets.
    train_X, train_y = split_xy(train, target)
    test_X, test_y = split_xy(test, target)

    # Build classifier.
    clf = DecisionTree(max_depth=args.max_depth)
    clf = clf.fit(train_X, train_y)
    print 'Accuracy: %.5f' % clf.score(test_X, test_y)

    # Compute RMSE
    predicted = clf.predict(test_X)
    print 'RMSE:     %.5f' % rmse(predicted, test_y)

    # Write out decision tree.
    with open('cs-dec-tree.dot', 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

