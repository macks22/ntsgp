import os
from scaffold import *
from ipr.ipr import IPR


def make_ipr_parser():
    parser = base_parser()
    parser.add_argument(
        '-k', '--nmodels',
        type=int, default=3,
        help='number of linear regression models')
    parser.add_argument(
        '-lw', '--lambda-w',
        type=float, default=0.01,
        help='regularization multiplier for P and W')
    parser.add_argument(
        '-lb', '--lambda-b',
        type=float, default=0.001,
        help='regularization multiplier for s and c')
    parser.add_argument(
        '--bounds', default='0,4',
        help='upper,lower bound for rating bounding')
    parser.add_argument(
        '-lr', '--lrate',
        type=float, default=0.001,
        help='learning rate')
    parser.add_argument(
        '-i', '--iters',
        type=int, default=10,
        help='number of iterations')
    parser.add_argument(
        '-e', '--epsilon',
        type=float, default=0.0001,
        help='stopping threshold for early stopping test')
    parser.add_argument(
        '-s', '--init-std',
        type=float, default=(1. / 2 ** 4),
        help='standard deviation of Gaussian noise used in model param'
             'initialization')
    parser.add_argument(
        '-n', '--nonneg',
        action='store_true', default=False,
        help='enable non-negativity constraints on all params')
    parser.add_argument(
        '-f', '--feature-guide',
        default='',
        help='file to specify target, categorical, and real-valued features; '
             'see the docstring for more detailed info on the format')
    return parser


if __name__ == "__main__":
    args = setup(make_ipr_parser)
    data = pd.read_csv(args.data_file).sort(['sid', 'termnum'])

    def eval_ipr(train, test):
        model = IPR(k=args.nmodels,
                    lambda_w=args.lambda_w,
                    lambda_b=args.lambda_b,
                    iters=args.iters,
                    std=args.init_std,
                    nonneg=args.nonneg,
                    verbose=args.verbose,
                    lrate=args.lrate,
                    epsilon=args.epsilon)

        saved_test = test.copy()

        eids, X, y, test_eids, test_X, test_y, f_indices, nb = \
            model.preprocess(train, test, args.feature_guide)

        print(eids.shape)
        print(X.shape)

        model.fit(X, y, eids, nb)

        # save fitted model params
        termnum = np.unique(test['termnum'])[0]
        name = 'ipr-model-term%d-tr%d-te%d' % (
            termnum, train.shape[0], test.shape[0])
        savedir = os.path.join('ipr-saves', '-'.join([name, model.args_suffix]))

        try:
            model.save(savedir)
        except OSError:
            pass

        # calculate and save feature importances
        I, I_pprof = model.feature_importance(
            X, y, eids, nb, train, f_indices, args.feature_guide)

        iname = os.path.join(savedir, 'imp') + '.csv'
        ipname = os.path.join(savedir, 'imp_pp') + '.csv'

        I.to_csv(iname)
        I_pprof.to_csv(ipname)

        with open(os.path.join(savedir, 'test-count'), 'w') as f:
            f.write('%s' % saved_test.shape[0])

        # return predictions
        saved_test[model.target] = model.predict(test_X, test_eids, nb)
        return saved_test


    results = method_error(data, eval_ipr, True, predict_cold_start=args.cold_start)
    by = args.plot if args.plot else ('cs' if args.cold_start else 'termnum')
    evaluation = eval_results(results, by=by)
    print evaluation

    if args.plot == 'pred':
        g1, g2 = plot_predictions(results)
    elif args.plot in ['termnum', 'sterm', 'cohort']:
        ax1, ax2 = plot_error_by(args.plot, results)
