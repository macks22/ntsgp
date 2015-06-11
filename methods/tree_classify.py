import logging
import argparse

import pandas as pd
from sklearn import ensemble, metrics, multiclass, preprocessing

from scaffold import *


def print_eval(test, predicted, num_labels, average='binary'):
    print '-' * 40
    print pd.crosstab(test, predicted, rownames=['True'],
                      colnames=['Predicted'], margins=True)
    print '-' * 40

    args = (test, predicted)
    kwargs = dict(labels=range(num_labels), average=average)
    print 'F1:        %.4f' % metrics.f1_score(*args, **kwargs)
    print 'Precision: %.4f' % metrics.precision_score(*args, **kwargs)
    print 'Recall:    %.4f' % metrics.recall_score(*args, **kwargs)

    nlabels = len(np.unique(test))
    if num_labels == 2 and nlabels == 2:
        print 'ROC AUC:   %.4f' % metrics.roc_auc_score(*args)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_file', action='store',
        default='../data/preprocessed-data-l1.csv')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False)
    parser.add_argument(
        '--method',
        choices=('rf', 'et'), default='rf',
        help='rf=random forest, et=extra trees')
    parser.add_argument(
        '-m', '--max-depth',
        type=int, default=10)
    parser.add_argument(
        '-n', '--n-estimators',
        type=int, default=100)
    parser.add_argument(
        '-nj', '--njobs',
        type=int, default=4)
    return parser


def decompose(df, k=20):
    """We assume the df has 3 columns: (sid, cid, grdpts). Decompose this using
    SVD and return the transformed subspace representation.
    """
    return df


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

    terms = list(sorted(data['termnum'].unique()))
    for termnum in terms:
        logging.info("making predictions for termnum %d" % termnum)

        train, test = split_train_test(data, termnum)
        test = remove_cold_start(train, test)
        if len(test) == 0:
            continue

        # # Decompose for subspace features
        # ucg = ['sid', 'cid', 'grdpts']
        # ucg_train = train[ucg]
        # ucg_test = test[ucg]

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
            method = (ensemble.RandomForestClassifier if args.method == 'rf'
                      else ensemble.ExtraTreesClassifier)

            clf = method(
                n_estimators=args.n_estimators, max_depth=args.max_depth,
                n_jobs=args.njobs)
            predictions = clf.fit(train_X, train_y,
                                  sample_weight=balance_weights(train_y))\
                             .predict(test_X)
            average = 'binary'
        else:
            raise NotImplementedError(
                "multi-class classification not implemented")
            sys.exit()
            # clf = multiclass.OneVsRestClassifier(
            #         estimator=ensemble.RandomForestClassifier(
            #             n_estimators=args.n_estimators,
            #             max_depth=args.max_depth, n_jobs=args.njobs))
            # predictions = clf.fit(train_X, train_y).predict(test_X)
            # average = 'micro'

        print '\nTERM %d:' % termnum
        print_eval(test_y, predictions, nlabels, average=average)

        df = test[tokeep].copy()
        df['predicted'] = predictions
        df['test'] = test_y
        results = pd.concat((results, df))

    print '\nTOTALS'
    # print_eval(all_test, all_predictions, num_labels, average)
    print_eval(results['test'], results['predicted'], num_labels, average)
