import sys
import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing

sys.path.append('../')
from recpipe import PreprocessedData


def split_xy(data, target='GRADE'):
    return data.drop(target, axis=1).values, data[target].values

def split_train_test(data, termnum):
    return data[data.termnum < termnum], data[data.termnum == termnum]

def train_test_for_term(data, termnum, target='grdpts'):
    train, test = split_train_test(data, 11)

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
    sqerror = (predicted - actual) ** 2
    mse = sqerror.sum() / len(sqerror)
    return np.sqrt(mse)


def read_data(fname):
    """Read in necessary columns from data."""
    logging.info("reading data from: %s" % fname)

    # Only read needed columns.
    task = PreprocessedData()
    cols = ['sid', 'cid', 'grdpts', 'termnum']
    features = list(set(task.rvals + cols))

    data = pd.read_csv(fname, usecols=features)
    return data.sort(['sid', 'termnum'])


def compute_rmse(clf, data, termnum, target='grdpts'):
    train_X, train_y, test_X, test_y = train_test_for_term(
        data, termnum, target)
    clf = clf.fit(train_X, train_y)

    # Compute RMSE
    predicted = clf.predict(test_X)
    return _rmse(predicted, test_y)


def rmse(predicted, actual, userid='sid', itemid='cid', value='grdpts'):
    """Compute root mean squared error between the predicted values and the
    actual values. This assumes some values are missing and only incorporates
    error measurements from the values present in the actual values.

    `actual` and `predicted` should be DataFrame objects with feature vectors
    for each data instance. The rmse is computed on the column named by `value`.
    """
    name_x = '%s_x' % value
    name_y = '%s_y' % value
    to_eval = actual.merge(predicted, how='left', on=[userid, itemid])
    to_eval = to_eval[[name_x, name_y]]  # filter down only to values

    # Sanity check; we don't want nan values here; that indicates no predictions
    # were made for some records.
    if to_eval.isnull().sum().sum() > 0:
        raise ValueError("predictions must be made for all missing values")

    # Compute error.
    error = (to_eval[name_x] - to_eval[name_y]).values
    mse = (error ** 2).sum() / len(error)
    return np.sqrt(mse)


def remove_cold_start(train, test, userid='sid', itemid='cid'):
    """Remove users/items from the test set that are not in the training set.
    """
    for key in [userid, itemid]:
        diff = np.setdiff1d(test[key].values, train[key].values)
        logging.info(
            "removing %d %s ids from the test set: %s" % (
                len(diff), key, ' '.join(map(str, diff))))
        cold_start = test[key].isin(diff)
        test = test[~cold_start]

    return test


def eval_method(data, method, dropna=False, *args, **kwargs):
    """Evaluate a particular baseline method `method` on the next-term
    prediction task with the given `data`. We assume the `termnum` column is
    present in the data and make predictions for each term by using all previous
    terms as training data. Additional argument will be passed to the `method`
    func.

    """
    results = {}  # key=termnum, val={'count': #, 'rmse': #}
    data = data.dropna() if dropna else data
    for termnum in sorted(data['termnum'].unique()):
        logging.info("making predictions for termnum %d" % termnum)
        train = data[data['termnum'] < termnum]
        test = data[data['termnum'] == termnum].copy()
        test = remove_cold_start(train, test)
        if len(test) == 0 or len(train) == 0:
            results[termnum] = {'count': 0, 'rmse': 0}
            continue

        to_predict = test.copy()
        to_predict['grdpts'] = np.nan
        predictions = method(train, to_predict, *args, **kwargs)
        term_rmse = rmse(predictions, test)
        results[termnum] = {'count': len(test), 'rmse': term_rmse}

    sqerror = sum((result['rmse'] ** 2) * result['count']
                  for result in results.values()
                  if result['count'] > 0)
    final_count = sum(result['count'] for result in results.values())
    final_rmse = np.sqrt(sqerror / final_count)
    results['all'] = {'count': final_count, 'rmse': final_rmse}
    return results

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
