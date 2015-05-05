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
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Read data.
    data = read_data(args.data_file)
    data = data.dropna()

    models = {
        'r': {0: tree.DecisionTreeRegressor,
              1: ensemble.RandomForestRegressor},
        'c': {0: tree.DecisionTreeClassifier,
              1: ensemble.RandomForestClassifier}
    }

    model_class = models[args.task][bool(args.forest)]
    params = {'max_depth': args.max_depth}
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
    print 'RMSE: %.5f' % eval_method(data, clf, True)['all']['rmse']

    # Write out decision tree.
    # if not args.forest:
    #     with open('cs-dec-tree.dot', 'w') as f:
    #         f = tree.export_graphviz(clf, out_file=f)

