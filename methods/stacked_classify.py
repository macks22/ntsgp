import logging
import argparse

import pandas as pd
from sklearn import tree, ensemble, metrics, preprocessing, linear_model, svm

from scaffold import *


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_file', action='store',
        default='../data/preprocessed-data-l1.csv')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False)
    parser.add_argument(
        '-l', '--loss',
        choices=('hinge', 'log', 'modified_huber', 'perceptron'),
        default='hinge',
        help='loss function to minimize')
    parser.add_argument(
        '-a', '--alpha',
        type=float, default=0.0001,
        help='regularization constant')
    parser.add_argument(
        '-p', '--penalty',
        choices=('l1', 'l2', 'elasticnet'), default='l2',
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

    logging.basicConfig(
        format='[%(asctime)s]: %(message)s',
        level=logging.INFO if args.verbose else logging.CRITICAL)

    data = pd.read_csv(args.data_file).sort(['sid', 'termnum'])

    # Missing value imputation for real-valued attributes.
    rvals = ['termnum', 'cohort', 'age', 'hsgpa', 'sat', 'chrs', 'clevel',
             'lterm_gpa', 'lterm_cum_gpa', 'total_chrs', 'num_enrolled',
             'lterm_cgpa', 'lterm_cum_cgpa', 'total_enrolled', 'term_chrs']
    for rval in rvals:
        data[rval] = data[rval].fillna(data[rval].median())

    print 'Number of records before discarding nan values: %d' % len(data)
    tokeep = ['sid', 'cid', 'alevel', 'major', 'sterm']
    data = data[rvals + tokeep].dropna()
    print 'Number of records after: %d' % len(data)

    num_labels = len(data.alevel.unique())
    results = pd.DataFrame()

    njobs=4

    clfs = [
        ensemble.AdaBoostClassifier(
            n_estimators=80, algorithm='SAMME', learning_rate=0.85,
            base_estimator=tree.DecisionTreeClassifier(max_depth=1)),
        ensemble.RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=njobs)
        # svm.SVC(
        #     kernel='sigmoid', C=0.9, gamma=0.01, class_weight='auto',
        #     cache_size=10000)
    ]
    bclf = linear_model.SGDClassifier(
        loss=args.loss, penalty=args.penalty, n_iter=args.niter,
        alpha=args.alpha, eta0=args.lrate,
        class_weight='auto', n_jobs=njobs,
        verbose=1 if args.verbose == 2 else 0)
    # bclf = ensemble.AdaBoostClassifier(
    #     n_estimators=256, algorithm='SAMME', learning_rate=1.5,
    #     base_estimator=tree.DecisionTreeClassifier(max_depth=1))

    terms = np.sort(data['termnum'].unique())
    for termnum in terms:
        logging.info("making predictions for termnum %d" % termnum)

        train, test = split_train_test(data, termnum)
        test = remove_cold_start(train, test)
        if len(test) == 0:
            continue

        # Split up predictors/targets.
        target = 'alevel'
        train_X, train_y = split_xy(train, target)
        test_X, test_y = split_xy(test, target)
        scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        # How many labels are we actually working with?
        uniq_labels = np.unique(train_y)
        nlabels = len(uniq_labels)
        if nlabels == 1:
            print 'only one label for term %d' % termnum
            continue
        elif nlabels == 2:
            blend_train = np.zeros((train_X.shape[0], len(clfs) * 2))
            blend_test = np.zeros((test_X.shape[0], len(clfs) * 2))
            weights = balance_weights(train_y)

            for num, clf in enumerate(clfs):
                logging.info('Training classifier [%d]' % num)
                clf = clf.fit(train_X, train_y, sample_weight=weights)
                blend_train[:, num:num + 2] = clf.predict_proba(train_X)
                blend_test[:, num:num + 2] = clf.predict_proba(test_X)

            predictions = bclf.fit(blend_train, train_y, sample_weight=weights)\
                              .predict(blend_test)
        else:
            raise NotImplementedError(
                "multi-class classification not implemented")
            sys.exit()

        print '\nTERM %d:' % termnum
        print_eval(test_y, predictions)

        df = test[tokeep].copy()
        df['predicted'] = predictions
        df['test'] = test_y
        results = pd.concat((results, df))

    print '\nTOTALS'
    print_eval(results['test'], results['predicted'])
