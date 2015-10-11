"""
Calculation and plotting of aggregate and time-series importance metrics.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import seaborn as sns

from ipr import plot_pprof_imp, plot_imp


IMP = 'imp.csv'
IMP_PP = 'imp_pp.csv'


def read_count(fname):
    with open(fname) as f:
        text = f.read()
    return int(text.strip())


def read_frames_and_counts(dirname, which):
    dirpath = os.path.abspath(dirname)
    contents = [os.path.join(dirpath, name) for name in os.listdir(dirpath)]
    subdirs = [fname for fname in contents if os.path.isdir(fname)]

    pattern = 'term(\d{1,2})'
    matcher = re.compile(pattern)
    get_tnum = lambda name: int(matcher.search(name).groups()[0])
    tnums = [get_tnum(os.path.basename(name)) for name in subdirs]

    fnames = [os.path.join(sub, which) for sub in subdirs]
    dfs = [pd.read_csv(fname, index_col=0) for fname in fnames]

    cfnames = [os.path.join(sub, 'test-count') for sub in subdirs]
    counts = [read_count(fname) for fname in cfnames]

    return tnums, dfs, counts


def avg_overall_imps(dfs, counts):
    nterms = len(dfs)
    ntest = sum(counts)
    df = pd.DataFrame(index=dfs[0].index, columns=dfs[0].columns, dtype=float)

    for name in dfs[0].index:
        # get weighted average and store in new data frame
        df.ix[name] = sum(dfs[i].ix[name] * counts[i]
                          for i in range(nterms)) / ntest
    return df


def avg_pprof_imps(dfs, counts, colname="Importance"):
    nterms = len(dfs)
    ntest = sum(counts)

    df = dfs[0].copy()
    df[colname] = 0

    for _, name, model_num, _ in df.itertuples():
        mask = (df.Feature == name) & (df.Model == model_num)
        for i in range(nterms):
            fmask = (dfs[i].Feature == name) & (dfs[i].Model == model_num)
            df.loc[mask, colname] += dfs[i].loc[fmask, colname] * counts[i]
        df.loc[mask, colname] /= ntest

    return df


def imps_to_ts(tnums, dfs, counts, colname="Importance"):
    nterms = len(dfs)
    ntest = sum(counts)

    for i in range(nterms):
        dfs[i]["Term"] = tnums[i]

    df = pd.concat(dfs)
    top10 = dfs[0].groupby('Feature')\
                  .mean()\
                  .sort('Importance', ascending=False)\
                  .head(10)\
                  .index.values
    df = df[df.Feature.isin(top10)]
    return df


def plot_imp_ts(I_ts):
    sns.plt.figure()
    ax = sns.tsplot(data=I_ts, time="Term", unit="Model", condition="Feature",
                    value="Importance")
    ax.figure.show()
    return ax


if __name__ == "__main__":
    dirname = 'ipr-saves'
    tnums, dfs, counts = read_frames_and_counts(dirname, IMP_PP)

    I_pprof = avg_pprof_imps(dfs, counts)
    plot_pprof_imp(I_pprof)

    I_ts = imps_to_ts(tnums, dfs, counts)
    plot_imp_ts(I_ts)
    raw_input()

