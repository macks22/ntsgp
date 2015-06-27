import argparse
from sklearn import tree, ensemble
from scaffold import *


def make_parser():
    parser = base_parser('Run boosted decision tree regression on ers data')
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
    args = setup(make_parser)

    # Read data.
    tokeep = \
        ['grdpts', 'sid', 'cid', 'termnum', 'major', 'sterm', 'cohort', 'cs']
    tokeep += RVALS
    data = pd.read_csv(args.data_file, usecols=tokeep).sort(['sid', 'termnum'])

    # Build classifier.
    clf = sklearn_model(
        ensemble.AdaBoostRegressor,
        base_estimator=tree.DecisionTreeRegressor(max_depth=args.max_depth),
        n_estimators=args.n_estimators)

    results = method_error(data, clf, True, predict_cold_start=args.cold_start)
    by = args.plot if args.plot else ('cs' if args.cold_start else 'termnum')
    evaluation = eval_results(results, by=by)
    print evaluation

    if args.plot == 'pred':
        g1, g2 = plot_predictions(results)
    elif args.plot in ['termnum', 'sterm', 'cohort']:
        ax1, ax2 = plot_error_by(args.plot, results)
