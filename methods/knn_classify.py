import logging
import argparse

import pandas as pd
from sklearn import metrics, preprocessing, neighbors

from scaffold import *


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_file', action='store',
        default='../data/preprocessed-data-l1.csv')
    parser.add_argument(
        '-k', type=int, default=5,
        help='number of neighbors')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False)
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

    terms = list(sorted(data['termnum'].unique()))
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
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=args.k, weights='distance')
            predictions = clf.fit(train_X, train_y)\
                             .predict(test_X)
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
