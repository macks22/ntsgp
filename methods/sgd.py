import argparse
from sklearn import linear_model, preprocessing
from scaffold import *


def make_parser():
    parser = argparse.ArgumentParser(
        description='Run decision tree regression/classification on ers data')
    parser.add_argument(
        'data_file', action='store',
        help='data file')
    parser.add_argument(
        '-l', '--loss',
        choices=('squared_loss', 'huber', 'epsilon_insensitive',
                 'squared_epsilon_insensitive'),
        default='squared_loss',
        help='loss function to minimize')
    parser.add_argument(
        '-a', '--alpha',
        type=float, default=0.0001,
        help='regularization constant')
    parser.add_argument(
        '-p', '--penalty',
        choices=('l1', 'l2', 'elasticnet'), default='l1',
        help='type of regularization penalty to use')
    parser.add_argument(
        '-n', '--niter',
        type=int, default=5,
        help='number of iterations to run')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.01,
        help='learning rate')
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
        linear_model.SGDRegressor, loss=args.loss, penalty=args.penalty,
        alpha=args.alpha, n_iter=args.niter, eta0=args.lrate)
    print 'RMSE: %.5f' % eval_method(data, clf, True)['all']['rmse']
