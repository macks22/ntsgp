import os
import numpy as np


def write_libfm(f, df, target='grdpts', cvals=None, rvals=None):
    """Write feature vectors in libFM format from a DataFrame. libFM uses the
    same format as libSVM. It takes feature vectors of the form:

        <target> <cval1>:1 <cval2>:1 ... <cvaln>:1 <rval1>:%f ... <rvaln>:%f

    After writing the target value (which is written as a float), we write all
    the features for the vector.

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
    # Sanity checking; make sure we have some values to write.
    if cvals is None:
        if rvals is None:
            raise ValueError("must pass either cvals or rvals")
        else:
            allvals = rvals
    else:
        if rvals is None:
            allvals = cvals
        else:
            allvals = cvals + rvals

    # More sanity checking; make sure all columns passed are in DataFrame.
    cols = df.columns
    if target not in cols:
        raise KeyError("target: %s not in DataFrame" % target)
    for colname in allvals:
        if colname not in cols:
            raise KeyError("colname %s not in DataFrame" % colname)

    # First, let's update the values for all cvals.
    max_idx = 0
    for cval in cvals:
        df[cval] += max_idx
        max_idx = np.nanmax(df[cval].values)

    max_idx += 1  # start value for rvals
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
        pieces = pieces + cval_part + rval_part
        return ' '.join(pieces)

    # TODO: consider adding chunksize param to reduce memory overhead.
    lines = df.apply(extract_row, axis=1)
    f.write('\n'.join(lines))



def write_triples(f, data, userid='sid', itemid='cid', rating='grdpts'):
    """Write a data file of triples (sparse matrix).

    :param str userid: Name of user id column (matrix rows).
    :param str itemid: Name of item id column (matrix cols).
    :param str rating: Name of rating column (matrix entries).
    """
    cols = [userid, itemid, rating]
    data.to_csv(f, sep='\t', header=False, index=False, columns=cols)
