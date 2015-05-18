import argparse
from sklearn import tree, ensemble
from scaffold import *


def make_parser():
    parser = argparse.ArgumentParser(
        description='Run decision tree regression/classification on ers data')
    parser.add_argument(
        'data_file', action='store',
        help='data file')
    parser.add_argument(
        '-n', '--n-estimators',
        action='store', type=int, default=10,
        help='# estimators to use')
    parser.add_argument(
        '-m', '--max-depth',
        type=int, default=4,
        help='max depth to grow the tree to')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Read data.
    data = read_data(args.data_file)
    data = data.dropna()
    if 'GRADE' in data:
        del data['GRADE']

    # Build classifier.
    clf = sklearn_model(
        ensemble.AdaBoostRegressor,
        base_estimator=tree.DecisionTreeRegressor(max_depth=args.max_depth),
        n_estimators=args.n_estimators)
    print 'RMSE: %.5f' % eval_method(data, clf, True)['all']['rmse']
