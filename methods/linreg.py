import argparse
from sklearn import linear_model
from scaffold import *


def make_parser():
    parser = base_parser('Run OLS regression on ers data')
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
    args = setup(make_parser)

    # Read data.
    tokeep = \
        ['grdpts', 'sid', 'cid', 'termnum', 'major', 'sterm', 'cohort', 'cs']
    tokeep += RVALS
    data = pd.read_csv(args.data_file, usecols=tokeep).sort(['sid', 'termnum'])

    # Build classifier.
    if args.type == 'lr':
        model_class = linear_model.LinearRegression
        params = {}
    else:
        model_class = linear_model.Ridge
        params = dict(alpha=args.alpha)

    clf = sklearn_model(
        model_class, **params)

    results = method_error(data, clf, True, predict_cold_start=args.cold_start)
    by = args.plot if args.plot else ('cs' if args.cold_start else 'termnum')
    evaluation = eval_results(results, by=by)
    print evaluation

    if args.plot == 'pred':
        g1, g2 = plot_predictions(results)
    elif args.plot in ['termnum', 'sterm', 'cohort']:
        ax1, ax2 = plot_error_by(args.plot, results)
