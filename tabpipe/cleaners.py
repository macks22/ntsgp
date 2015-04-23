"""
Cleaning functions.

"""
import pandas as pd
import numpy as np
from collections import Counter


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
