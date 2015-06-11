import sys
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../')
sys.path.append('/home/msweene2/ers-data/')


def split_xy(data, target='GRADE'):
    return data.drop(target, axis=1).values, data[target].values

def split_train_test(data, termnum):
    return data[data.termnum < termnum], data[data.termnum == termnum]

def train_test_for_term(data, termnum, target='grdpts', cold_start=True):
    train, test = split_train_test(data, termnum)

    if not cold_start:
        test = remove_cold_start(train, test)

    if len(test) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Split up predictors/targets.
    train_X, train_y = split_xy(train, target)
    test_X, test_y = split_xy(test, target)
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    return train_X, train_y, test_X, test_y

def _rmse(predicted, actual):
    """Compute root mean squared error between the predicted values and the
    actual values. This assumes `predicted` and `actual` are arrays of values to
    compare, they are the same length, and there are no missing values.
    """
    return np.sqrt(((predicted - actual) ** 2).sum() / len(predicted))

def _mae(predicted, actual):
    return abs((predicted - actual)).sum() / len(predicted)


def read_data(fname, usecols=['sid', 'cid', 'grdpts', 'termnum']):
    """Read in necessary columns from data."""
    logging.info("reading data from: %s" % fname)

    # Only read needed columns.
    features = list(set(task.rvals + usecols))

    data = pd.read_csv(fname, usecols=features)
    return data.sort(['sid', 'termnum'])


def compute_rmse(clf, data, termnum, target='grdpts'):
    train_X, train_y, test_X, test_y = train_test_for_term(
        data, termnum, target)
    clf = clf.fit(train_X, train_y)

    # Compute RMSE
    predicted = clf.predict(test_X)
    return _rmse(predicted, test_y)


def _compute_error(predicted, actual, userid='sid', itemid='cid',
                  value='grdpts', bounds=(0, 4)):
    # Bound predictions.
    low, high = bounds
    predicted[predicted[value] > high][value] = high
    predicted[predicted[value] < low][value] = low

    name_x = '%s_x' % value
    name_y = '%s_y' % value
    to_eval = actual.merge(predicted, how='left', on=[userid, itemid])
    to_eval = to_eval[[name_x, name_y]]  # filter down only to values

    # Sanity check; we don't want nan values here; that indicates no predictions
    # were made for some records.
    if to_eval.isnull().sum().sum() > 0:
        raise ValueError("predictions must be made for all missing values")

    return (to_eval[name_x] - to_eval[name_y]).values


def rmse(predicted, actual, userid='sid', itemid='cid', value='grdpts'):
    """Compute root mean squared error between the predicted values and the
    actual values. This assumes some values are missing and only incorporates
    error measurements from the values present in the actual values.

    `actual` and `predicted` should be DataFrame objects with feature vectors
    for each data instance. The rmse is computed on the column named by `value`.
    """
    error = _compute_error(predicted, actual, userid, itemid, value)
    mse = (error ** 2).sum() / len(error)
    return np.sqrt(mse)

def mae(predicted, actual, userid='sid', itemid='cid', value='grdpts'):
    error = _compute_error(predicted, actual, userid, itemid, value)
    return abs(error).sum() / len(error)


def remove_cold_start(train, test, userid='sid', itemid='cid'):
    """Remove users/items from test set that are not in the train set."""
    for key in [userid, itemid]:
        diff = np.setdiff1d(test[key].values, train[key].values)
        logging.info(
            "removing %d %s ids from the test set." % (len(diff), key))
        logging.debug(' '.join(map(str, diff)))
        cold_start = test[key].isin(diff)
        test = test[~cold_start]

    return test


def run_method(data, method, dropna=False, *args, **kwargs):
    """Evaluate a particular baseline method `method` on the next-term
    prediction task with the given `data`. We assume the `termnum` column is
    present in the data and make predictions for each term by using all previous
    terms as training data. Additional argument will be passed to the `method`
    func.

    """
    data = data.dropna() if dropna else data
    terms = list(sorted(data['termnum'].unique()))

    for termnum in terms:
        logging.info("making predictions for termnum %d" % termnum)
        # train_mask = ((data['termnum'] < termnum) &
        #               (data['termnum'] > termnum - 5))
        train_mask = data['termnum'] < termnum
        train = data[train_mask]
        test = data[data['termnum'] == termnum].copy()
        test = remove_cold_start(train, test)
        if len(test) == 0 or len(train) == 0:
            yield (termnum, np.array([]), np.array([]))
            continue

        to_predict = test.copy()
        to_predict['grdpts'] = np.nan
        predictions = method(train, to_predict, *args, **kwargs)
        yield (termnum, predictions, test)


def method_error(data, method, dropna=False, *args, **kwargs):
    results = pd.DataFrame()
    evaluator = run_method(data, method, dropna, *args, **kwargs)
    for termnum, predictions, test in evaluator:
        if not len(predictions) == 0 and not len(test) == 0:
            keep_cols= ['sid', 'cid', 'termnum', 'major', 'sterm', 'grdpts']
            df = test[keep_cols].copy()
            df['pred'] = predictions['grdpts']
            df['error'] = _compute_error(predictions, test)
            results = pd.concat((results, df))

    return results


def eval_results(results, by='termnum'):
    """Given results which include the error for each prediction, return an
    evaluation in terms of (1) RMSE, (2) MAE, (3) record count.
    """
    rmse = results.groupby(by).apply(
        lambda df: np.sqrt((df['error'].values ** 2).sum() / len(df)))
    mae = results.groupby(by).apply(
        lambda df: abs(df['error'].values).sum() / len(df))
    mae_std = results.groupby(by).apply(
        lambda df: abs(df['error'].values).std())
    counts = results.groupby(by)['error'].count()

    total_count = len(results)
    rmse['all'] = np.sqrt((rmse**2 * counts).sum() / total_count)
    mae['all'] = (mae * counts).sum() / total_count
    mae_std['all'] = abs(results['error'].values).std()
    counts['all'] = total_count

    results = pd.DataFrame()
    results['rmse'] = rmse
    results['mae'] = mae
    results['mae_std'] = mae_std
    results['counts'] = counts
    return results


def df_rmse(df):
    return (df['error'] ** 2).sum() / len(df)

def df_mae(df):
    return abs(df['error']).sum() / len(df)

def granular_eval(results, metric=df_rmse, metric_name='RMSE'):
    """Plot a granular view of the error for a particular result. This takes the
    form of a heatmap (termnum X sterm) with a barplot on the top and side for
    by-termnum and by-sterm error.
    """
    mat = results.groupby(['termnum', 'sterm']).apply(metric).unstack(1)
    by_sterm = results.groupby('sterm').apply(metric)
    by_term = results.groupby('termnum').apply(metric)

    # Set up plot geometry.
    f = plt.figure(figsize=(9, 9))
    gs = plt.GridSpec(15, 6)
    top_hist_ax = f.add_subplot(gs[:5, :4])
    map_ax = f.add_subplot(gs[5:-2, :4])
    bar_ax = f.add_subplot(gs[-1, :4])
    right_hist_ax = f.add_subplot(gs[5:-2, 4:])

    # Plot top histogram (sterm).
    num_sterms = len(results['sterm'].unique())
    top_hist_ax.bar(range(num_sterms), by_sterm, 1, ec='w', lw=2, color='.3',
                    alpha=0.7)
    top_hist_ax.set(xticks=[], ylabel='%s by Student Term' % metric_name)

    # Plot right histogram (termnum).
    num_terms = len(results['termnum'].unique())
    right_hist_ax.barh(range(num_terms), by_term, height=1, ec='w', lw=2,
                       color='.3', alpha=0.7)
    right_hist_ax.set(yticks=[], xlabel='%s by Term Number' % metric_name)
    right_hist_ax.set_ylim(0, num_terms)
    right_hist_ax.invert_yaxis()
    right_hist_ax.xaxis.tick_top()
    right_hist_ax.xaxis.set_label_position('top')

    # Plot heatmap and heat bar.
    sns.heatmap(mat, cmap='Reds', ax=map_ax, cbar_ax=bar_ax,
                cbar_kws={'orientation': 'horizontal'})
    bar_ax.set(xlabel=metric_name)

    sns.plt.show()
    return f


# TODO: DEPRECATE
def eval_method(data, method, dropna=False, *args, **kwargs):
    """Evaluate a particular baseline method `method` on the next-term
    prediction task with the given `data`. We assume the `termnum` column is
    present in the data and make predictions for each term by using all previous
    terms as training data. Additional argument will be passed to the `method`
    func.

    """
    results = {}  # key=termnum, val={'count': #, 'rmse': #}
    evaluator = run_method(data, method, dropna, *args, **kwargs)
    for termnum, predictions, test in evaluator:
        if len(predictions) == 0 or len(test) == 0:
            results[termnum] = {'count': 0, 'rmse': 0.0, 'mae': 0.0}
        else:
            results[termnum] = {
                'count': len(test),
                'rmse': rmse(predictions, test),
                'mae': mae(predictions, test)
            }

    sqerror = sum((result['rmse'] ** 2) * result['count']
                  for result in results.values()
                  if result['count'] > 0)
    abserror = sum(result['mae'] * result['count']
                   for result in results.values()
                   if result['count'] > 0)
    final_count = sum(result['count'] for result in results.values())
    final_rmse = np.sqrt(sqerror / final_count)
    final_mae = abserror / final_count

    results['all'] = {
        'count': final_count,
        'rmse': final_rmse,
        'mae': final_mae
    }
    return results


def plot_predictions(results):
    """Plot the predictions made vs. the actual grades.
    """
    results = results.rename(columns={'termnum': 'term'})
    results = results[['term', 'pred', 'grdpts']].copy()

    results['predicted'] = 1
    actual = results.copy()
    actual['predicted'] = 0

    results['grdpts'] = results['pred']
    del results['pred']
    del actual['pred']

    data = pd.concat((results, actual))

    def plot(df, max_term, min_term=0):
        data = df[(df.term <= max_term) & (df.term >= min_term)]
        grid = sns.FacetGrid(data, row='term', col='predicted',
                             margin_titles=True, sharey=False)
        grid.map(plt.hist, 'grdpts')
        sns.plt.show()
        return grid

    g1 = plot(data, max_term=7)
    g2 = plot(data, min_term=8, max_term=14)
    return g1, g2


def _plot_error(error, labels, title="Actual - Predicted", xlabel="Term Number",
                ylabel="Error"):
    ax1 = sns.violinplot(error, names=labels, figsize=(12, 6))
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    sns.plt.show()

    ax2 = sns.boxplot(error, names=labels)
    ax2.set_title(title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    sns.plt.show()

    return ax1, ax2


def plot_error_by(by, results):
    """Take results from prediction, which should include an error column and
    plot by the sterm.

    """
    tups = [(val, results[results[by] == val]['error'].values)
            for val in np.sort(results[by].unique())]
    names, errors = zip(*tups)
    return _plot_error(errors, names, xlabel=by)


def quiet_delete(data, col):
    try:
        return data.drop(col, axis=1)
    except:
        return data

def sklearn_model(model_class, *args, **kwargs):
    # Change target column if requested.
    value = kwargs.get('_value', 'grdpts')
    try: del kwargs['_value']
    except: pass

    def model(train, test, value=value):
        to_predict = test.copy()
        logging.info(
            'predicting %d values using %s' % (
                len(test), model_class.__name__))
        clf = model_class(*args, **kwargs)

        for feature in ['sid', 'cid']:
            train = quiet_delete(train, feature)
            to_predict = quiet_delete(to_predict, feature)

        # Split up predictors/targets.
        train_X, train_y = split_xy(train, value)
        test_X, test_y = split_xy(to_predict, value)
        scaler = preprocessing.StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        # Learn model and make predictions.
        clf = clf.fit(train_X, train_y)
        predicted = clf.predict(test_X)
        to_predict[value] = predicted

        to_predict = to_predict.merge(
            test[['sid', 'cid']], how='left', left_index=True, right_index=True)
        return to_predict

    return model


def base_parser(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'data_file', action='store')
    parser.add_argument(
        '-v', '--verbose', type=int, default=0,
        help='enable verbose logging output')
    parser.add_argument(
        '--plot', action='store',
        choices=('term', 'pred', 'sterm'),
        default='')
    return parser


def setup(parser_func):
    parser = parser_func()
    args = parser.parse_args()

    level = (logging.DEBUG if args.verbose == 2 else
             logging.INFO if args.verbose == 1 else
             logging.ERROR)
    logging.basicConfig(
        level=level,
        format='[%(asctime)s][%(levelname)s]: %(message)s')

    return args


def read_some_data(data_file, cvals=['sid', 'cid', 'sterm', 'grdpts']):
    # All real-valued attributes of interest.
    rvals = ['termnum', 'cohort', 'age', 'hsgpa', 'sat', 'chrs', 'clevel',
             'lterm_gpa', 'lterm_cum_gpa', 'total_chrs', 'num_enrolled',
             'lterm_cgpa', 'lterm_cum_cgpa', 'total_enrolled', 'term_chrs']

    cols = cvals + rvals

    # Read data.
    data = pd.read_csv(data_file, usecols=cols).sort(['sid', 'termnum'])

    # Missing value imputation.
    for rval in rvals:
        data[rval] = data[rval].fillna(data[rval].median())

    # Drop records with NaN values after imputation.
    logging.info(
        'Number of records before discarding nan values: %d' % len(data))
    data = data.dropna()
    logging.info('Number of records after: %d' % len(data))
    return data


def balance_weights(y):
    """Compute sample weights such that the class distribution of y becomes
    balanced. In other words, count the number of each class and weight the
    instances by (1 / # instances for class) * min(# instances for each class).

    Parameters
    ----------
    y : array-like
        Labels for the samples.
    Returns
    -------
    weights : array-like
        The sample weights.
    """
    y = np.asarray(y)
    y = np.searchsorted(np.unique(y), y)
    bins = np.bincount(y)

    weights = 1. / bins.take(y)
    weights *= bins.min()
    return weights


def print_eval(test, predicted):
    print '-' * 40
    print pd.crosstab(test, predicted, rownames=['True'],
                      colnames=['Predicted'], margins=True)
    print '-' * 40

    args = (test, predicted)
    kwargs = dict(labels=[0,1])
    print 'F1:        %.4f' % metrics.f1_score(*args, **kwargs)
    print 'Precision: %.4f' % metrics.precision_score(*args, **kwargs)
    print 'Recall:    %.4f' % metrics.recall_score(*args, **kwargs)

    uniq_labels = len(np.unique(test))
    if uniq_labels >= 2:
        print 'ROC AUC:   %.4f' % metrics.roc_auc_score(*args)
