import argparse
from sklearn import neighbors
from scaffold import *


def make_parser():
    parser = base_parser(
        'Run kNN regression on ers data')
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
    args = setup(make_parser)

    # Read data.
    tokeep = ['sid', 'cid', 'major', 'sterm', 'grdpts']
    data = read_some_data(args.data_file, tokeep)

    clf = sklearn_model(
        neighbors.KNeighborsRegressor, n_neighbors=args.k, weights=args.weights)

    results = method_error(data, clf, True)
    evaluation = eval_results(
        results, by='sterm' if args.plot == 'sterm' else 'termnum')
    print evaluation

    if args.plot == 'pred':
        g1, g2 = plot_predictions(results)
    if args.plot == 'term':
        ax1, ax2 = plot_error_by('termnum', results)
    elif args.plot == 'sterm':
        ax1, ax2 = plot_error_by('sterm', results)
