import argparse
from sklearn import linear_model
from scaffold import *


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
    data = read_data(args.data_file)
    data = data.dropna()
    if 'GRADE' in data:
        del data['GRADE']

    # Build classifier.
    if args.type == 'lr':
        model_class = linear_model.LinearRegression
        params = {}
    else:
        model_class = linear_model.Ridge
        params = dict(alpha=args.alpha)

    clf = sklearn_model(
        model_class, **params)
    print 'RMSE: %.5f' % eval_method(data, clf, True)['all']['rmse']
