import argparse
from sklearn import neighbors
from scaffold import *


def make_parser():
    parser = argparse.ArgumentParser(
        description='Run decision tree regression/classification on ers data')
    parser.add_argument(
        'data_file', action='store',
        help='data file')
    parser.add_argument(
        '-k', action='store',
        type=int, default=5,
        help='how many neighbors to use')
    parser.add_argument(
        '-w', '--weights',
        choices=('uniform', 'distance'), default='distance',
        help='choose uniform to weight all neighbors the same regardless of\
        distance')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Read data.
    data = read_data(args.data_file)
    if 'GRADE' in data:
        del data['GRADE']

    clf = sklearn_model(
        neighbors.KNeighborsRegressor, n_neighbors=args.k, weights=args.weights)
    print 'RMSE: %.5f' % eval_method(data, clf, True)['all']['rmse']
