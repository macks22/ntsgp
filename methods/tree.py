import argparse
from sklearn import tree, ensemble
from scaffold import *


def rf_gi(rf, data):
    dfs = []
    cols = data.drop('grdpts', axis=1).columns
    for tnum in np.sort(data.termnum.unique()):
        train_x, train_y, _, _ = train_test_for_term(
            data, tnum, 'grdpts')
        if len(train_x) == 0:
            continue

        rf = rf.fit(train_x, train_y)
        dfs.append(
            pd.DataFrame(zip(cols, rf.feature_importances_))\
              .rename(columns={0: 'features', 1: 'gi'})\
              .sort('gi', ascending=False)\
              .set_index('features')
        )
    return dfs


def make_parser():
    parser = base_parser('Run Random Forest regression on ers data')
    parser.add_argument(
        '-t', '--task', action='store', default='r',
        help='r=regression, c=classification')
    parser.add_argument(
        '-f', '--forest',
        action='store', type=int, default=0,
        help='# estimators to use; if passed, random forests are used')
    parser.add_argument(
        '-m', '--max-depth',
        type=int, default=4,
        help='max depth to grow the tree to')
    parser.add_argument(
        '-nj', '--njobs',
        type=int, default=4)
    parser.add_argument(
        '-o', '--out-tree',
        type=int, default=14,
        help='output dot file with decision tree learned on this termnum')
    return parser


if __name__ == "__main__":
    args = setup(make_parser)

    # Read data.
    tokeep = \
        ['grdpts', 'sid', 'cid', 'termnum', 'major', 'sterm', 'cohort', 'cs',
         'irank', 'itenure', 'iclass']
    tokeep += RVALS
    data = pd.read_csv(args.data_file, usecols=tokeep).sort(['sid', 'termnum'])

    models = {
        'r': {0: tree.DecisionTreeRegressor,
              1: ensemble.RandomForestRegressor},
        'c': {0: tree.DecisionTreeClassifier,
              1: ensemble.RandomForestClassifier}
    }

    model_class = models[args.task][bool(args.forest)]
    params = {'max_depth': args.max_depth, 'n_jobs': args.njobs}
    if args.forest:
        params['n_estimators'] = args.forest

    if args.task == 'r':
        if 'GRADE' in data:
            del data['GRADE']
        target = 'grdpts'
    else:
        del data['grdpts']
        target = 'GRADE'

    params['_value'] = target
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

    # Write out decision tree.
    if args.out_tree and not args.forest:
        for feature in ['sid', 'cid']:
            data = quiet_delete(data, feature)

        # Learn classifier for last term prediction.
        train_X, train_y, test_X, test_y = train_test_for_term(
            data, data.termnum.max(), target)

        del params['_value']
        clf = model_class(**params)
        clf = clf.fit(train_X, train_y)
        cols = data.drop(target, axis=1).columns
        with open('dec-tree.dot', 'w') as f:
            f = tree.export_graphviz(clf, out_file=f, feature_names=cols)

