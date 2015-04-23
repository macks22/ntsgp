"""
Functions to filter out rows from a table.

"""
import pandas as pd
import numpy as np


def filter_out_nan_values(df, colnames):
    """Filter out rows that have nan values in any of the given columns."""
    mask = pd.Series([False] * len(df))
    for colname in colnames:
        mask = mask | (df[colname].isnull())
    return df[mask].index


def filter_to_numeric_range(df, colnames, range):
    """Filter out rows that have values beyond the given range in any of the
    given columns. `range` is a tuple of (lower, upper) bound ints. To use
    this function, define a wrapper around it that calls this function with
    a fixed range.
    """
    lower, upper = range
    mask = pd.Series([False] * len(df))
    for colname in colnames:
        mask = mask | (df[colname] > upper) | (df[colname] < lower)
    return df[mask].index

