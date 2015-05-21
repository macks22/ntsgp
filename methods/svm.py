import argparse
from sklearn import svm
from scaffold import *


def make_parser():
    parser = argparse.ArgumentParser(
        description='Run decision tree regression/classification on ers data')
    parser.add_argument(
        'data_file', action='store',
        help='data file')
    parser.add_argument(
        '-k', '--kernel', action='store',
        choices=('linear', 'polynomial', 'rbf', 'sigmoid'),
        default='rbf',
        help='kernel function to use; see scikit-learn SVM docs for detail')
    parser.add_argument(
        '-g', '--gamma',
        type=float, default=0.0,
        help='kernel coefficient')
    parser.add_argument(
        '-p', '--penalty',
        type=float, default=1.0,
        help='penalty for the error term')
    parser.add_argument(
        '-d', '--degree',
        type=int, default=3,
        help='degree of kernel function; default 3')
    parser.add_argument(
        '-e', '--epsilon',
        type=float, default=0.1,
        help='specifies the epsilon-tube within which no penalty is associated\
        in the training loss function with points predicted within a distance\
        epsilon from the actual value')

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
    params = {
        'kernel': args.kernel,
        'C': args.penalty,
        'epsilon': args.epsilon,
        'cache_size': 5000,
        'gamma': args.gamma,
        'degree': args.degree
    }

    clf = sklearn_model(
        svm.SVR, **params)

    result = eval_method(data, clf, True)['all']
    print 'RMSE: %.5f' % result['rmse']
    print 'MAE:  %.5f' % result['mae']
