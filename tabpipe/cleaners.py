"""
Cleaning functions.

"""
from collections import Counter

import pandas as pd
import numpy as np
from scipy import stats


def fill_nan_with_mode(df, colname):
    """Fill all nan values in the column with the mode of that column."""
    nan_mask = df[colname].isnull()
    vals = df[colname][~nan_mask]
    counts = Counter(vals)
    mode = counts.most_common(1)[0][0]
    df[colname][nan_mask] = mode


def strip_text_column(df, colname):
    """Strip whitespace from a text column."""
    df[colname] = df[colname].str.strip()


def fill_nan_with_other_col(df, colname, othercol):
    """Fill nan values in `colname` with those in `othercol`."""
    nan_mask = df[colname].isnull()
    df[colname][nan_mask] = df[othercol][nan_mask]

def fill_nan_with_2col_1to1_map(df, colname, othercol):
    """Construct map from another column to the column of interest. Groupby the
    other column, find the mode value excluding nan values in the column of
    interest for each group. Then use these val: mode mappings to fill in the
    nan values.
    """
    nan_mask = df[colname].isnull()
    workframe = df[~nan_mask]
    mapping = workframe.groupby(othercol).agg(
        lambda s: stats.mode(s[colname])[0])[colname]

    # Fill nan values using mapping.
    fillvals = df[othercol].apply(lambda k: mapping[k])
    df[colname][nan_mask] = fillvals
