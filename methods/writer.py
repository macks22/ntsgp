import os
import numpy as np
import pandas as pd


def make_prev_crecord_fmter(data):
    """Create a function closure with a cache. Return the internal function
    which formats the previous course grades in libFM format.
    """
    cache = {}
    data['pcgrade'] = data[['pcid', 'grdpts']].apply(
        lambda s: '%d:%f' % tuple(s.values), axis=1)
    def prior_cgrades(sid, termnum):
        try:
            return cache[(sid, termnum)]
        except KeyError:
            df = data[(data.sid == sid) & (data.termnum < termnum)]
            if not len(df):
                cache[(sid, termnum)] = []
                return []
            string = ' '.join(df.pcgrade)
            cache[(sid, termnum)] = [string]
            return [string]

    return prior_cgrades


def write_libfm(ftrain, ftest, train, test, target='grdpts', userid='sid',
                itemid='cid', cvals=None, rvals=None, prev_cgrades=False):
    """Write feature vectors in libFM format from a DataFrame. libFM uses the
    same format as libSVM. It takes feature vectors of the form:

        <target> <cval1>:1 <cval2>:1 ... <cvaln>:1 <rval1>:%f ... <rvaln>:%f

    After writing the target value (which is written as a float), we write all
    the features for the vector.

    Note that we take both the train and the test frame as required because the
    max values in the indices must be calculated AT THE SAME TIME. Otherwise,
    the data files will end up having different indices for the same entities,
    which is garbage.

    There are two types of features we must be concerned with when writing these
    files: set-categoricals (cvals) and real-valued attributes (rvals).

    For cvals, we assume each has already been encoded using a suitable
    numerical encoding (contiguous from 0). We write all of these values first,
    using one-hot encoding. They are written in the order they are passed in the
    `cvals` parameter. The first cval starts at an id of 0. The second starts
    off from the max id of the last, and so on. The max id of the last cval (+1)
    is used as the starting id for the first rval.

    For rvals, we need to write both the id number and the actual value. Since
    we can encode all ints as floats, we simply use floats throughout. Each rval
    gets a single id. They are written in the order passed in the `rvals`
    parameter.
    """
    # Make sure we have data.
    if (len(test) == 0) or (len(train) == 0):
        return

    # Set up values, inserting user and item ids into the cvals.
    cvals = cvals if cvals else []
    cvals.insert(0, itemid)
    cvals.insert(0, userid)

    rvals = rvals if rvals else []
    allvals = cvals + rvals

    # Sanity check; make sure train/test columns are the same.
    train_cols = np.sort(train.columns.values)
    test_cols = np.sort(test.columns.values)
    if not (train_cols == test_cols).all():
        raise ValueError(
            "train and test columns do not match:\nTRAIN: %s\nTEST: %s" % (
                ','.join(map(str, train.columns)),
                ','.join(map(str, test.columns))))

    # More sanity checking; make sure all columns passed are in DataFrame.
    cols = train.columns  # we now know train/test have same columns
    if target not in cols:
        raise KeyError("target: %s not in DataFrame" % target)
    for colname in allvals:
        if colname not in cols:
            raise KeyError("colname %s not in DataFrame" % colname)

    # First, let's update the values for all cvals.
    max_idx = 0
    for cval in cvals:
        train[cval] += max_idx
        test[cval] += max_idx
        max_idx = max(
            np.nanmax(train[cval].values), np.nanmax(test[cval].values)) + 1

    # If requested, include ids for previous course grades.
    # This relies on the presence of a 'pcid' key for previous course id.
    if prev_cgrades:
        alldata = pd.concat((train, test))
        cid_range = alldata.cid.max() - alldata.cid.min()
        ids_between = max_idx - alldata.cid.max()
        diff = cid_range + ids_between
        alldata['pcid'] = alldata['cid'] + diff
        prior_cgrades = make_prev_crecord_fmter(alldata)
        max_idx += len(alldata.cid.unique()) + 1

    rval_indices = np.arange(len(rvals)) + max_idx

    # Now we need to actually extract the cvals.
    # This will be slow, because we must check each one for nan values.
    def extract_row(series):
        pieces = ['%f' % series[target]]
        cval_part = \
            ['%d:1' % series[cval] for cval in cvals if ~np.isnan(series[cval])]
        rval_part = \
            ['%d:%f' % (idx, series[key])
             for idx, key in zip(rval_indices, rvals) if ~np.isnan(series[key])]
        if prev_cgrades:
            pieces = (pieces + cval_part +
                      prior_cgrades(series['sid'], series['termnum']) +
                      rval_part)
        else:
            pieces = pieces + cval_part + rval_part
        return ' '.join(pieces)

    # TODO: consider adding chunksize param to reduce memory overhead.
    lines = train.apply(extract_row, axis=1)
    ftrain.write('\n'.join(lines))
    lines = test.apply(extract_row, axis=1)
    ftest.write('\n'.join(lines))


def write_triples(f, data, userid='sid', itemid='cid', rating='grdpts'):
    """Write a data file of triples (sparse matrix).

    :param str userid: Name of user id column (matrix rows).
    :param str itemid: Name of item id column (matrix cols).
    :param str rating: Name of rating column (matrix entries).
    """
    cols = [userid, itemid, rating]
    data.to_csv(f, sep='\t', header=False, index=False, columns=cols)
