import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from imps import read_frames_and_counts, avg_overall_imps


plt.style.use('ggplot')
cb_palette = sns.color_palette('colorblind')


def plot_ntvt(I_nt, I_tr, colname='Importance'):
    """Plot non-transfer vs. transfer flat importance.
    When the importance comes from multiple terms, it should be a an average
    weighted by the number of records predicted in each term.

    This function takes two data frames, both with the same format. The first
    should contain the importance metrics for non-transfer students, and the
    second should contain the metrics for transfer students. The index should be
    the feature names and `colname` should contain the importance metrics (which
    should sum to 1).

    TODO: it's not really necessary to have this graph be colored, since the
    color carries no meaning at this point. You could do a heatmap-type
    coloring, which would make the color a useful thing.
    """
    fig, ax = plt.subplots()
    idx = np.arange(I_nt.shape[0])
    lw = 1

    # Align features, sorted by the non-transfer feature importance
    for col in np.union1d(I_nt.index, I_tr.index):
        I_nt.ix[col] = I_nt.ix[col] if col in I_nt.index else 0.0
        I_tr.ix[col] = I_tr.ix[col] if col in I_tr.index else 0.0

    I_nt = I_nt.sort(colname, ascending=True)
    I_tr = I_tr.set_index(I_nt.index)

    vals1 = I_tr[colname] * 100
    vals2 = I_nt[colname] * 100

    rects1 = ax.barh(idx, vals1, lw, color=cb_palette[0])
    rects2 = ax.barh(idx, -vals2, lw, color=cb_palette[1])

    # get rectangle bounding boxes and calculate text positioning xs and ys
    bboxes = [rect.get_bbox() for rect in rects2]
    far_left = np.array([bb.xmin for bb in bboxes])

    heights = np.array([bb.height for bb in bboxes])
    ymaxes = np.array([bb.ymax for bb in bboxes])

    text_ys = ymaxes - 0.6 * heights
    xlims = ax.get_xlim()
    x_range = xlims[1] - xlims[0]
    spacer = x_range / 100
    text_xs = far_left - spacer

    # place labels directly next to left side of left-most bars
    for x, y, name in zip(text_xs, text_ys, I_nt.index):
        ax.text(x, y, name, va='center', ha='right',
                weight='bold', alpha=0.7)

    # remove yticks/grid and resize yaxis
    ax.set_yticks([])
    ax.yaxis.grid(False)

    # ax.set_ylim(0, max(yticks) + max(heights))
    ax.set_xlim(xlims[0] + 2 * spacer, xlims[1] - 2 * spacer)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    ax.set_ylim(ylims[0] - 1, max(ymaxes) + 3)
    ylims = ax.get_ylim()

    # Add top-level labels for "Transfer" and "Non-Transfer".
    # These specify which side of the graph is which.
    point0 = rects2[0].get_bbox().xmax
    x1 = point0 - 3 * spacer
    x2 = point0 + 3 * spacer
    ax.text(x1, ylims[1] - 2, 'Non-Transfer',
            fontsize='large', variant='small-caps', weight='bold',
            ha='right', va='bottom', color=cb_palette[1])
    ax.text(x2, ylims[1] - 2, 'Transfer',
            fontsize='large', variant='small-caps', weight='bold',
            ha='left', va='bottom', color=cb_palette[0])

    # add "%" to xticklabels
    xlabs = ax.get_xticks().tolist()
    ax.set_xticklabels(['%d%%' % abs(num) for num in xlabs])

    ax.set_title('Non-Transfer vs. Transfer Feature Importance Comparison')
    ax.set_xlabel('Percent of Absolute Deviation From Global Intercept')

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'non_transfer',
        help='path of dir with non-transfer importance metrics')
    parser.add_argument(
        'transfer',
        help='path of dir with transfer importance metrics')

    args = parser.parse_args()

    tnums, dfs, counts = read_frames_and_counts(args.non_transfer, 'imp.csv')
    I_nt = avg_overall_imps(dfs, counts)

    tnums, dfs, counts = read_frames_and_counts(args.transfer, 'imp.csv')
    I_tr = avg_overall_imps(dfs, counts)

    # ax = plot_ntvt(I_nt, I_tr)

    colname = "Importance"
    fig, ax = plt.subplots()
    idx = np.arange(I_nt.shape[0])
    lw = 1

    # Align features, sorted by the non-transfer feature importance
    for col in np.union1d(I_nt.index, I_tr.index):
        I_nt.ix[col] = I_nt.ix[col] if col in I_nt.index else 0.0
        I_tr.ix[col] = I_tr.ix[col] if col in I_tr.index else 0.0

    I_nt = I_nt.sort(colname, ascending=True)
    sortby = pd.Categorical(I_tr.index, I_nt.index)
    I_tr.index = sortby
    I_tr = I_tr.sort_index()

    vals1 = I_tr[colname] * 100
    vals2 = I_nt[colname] * 100

    rects1 = ax.barh(idx, vals1, lw, color=cb_palette[0])
    rects2 = ax.barh(idx, -vals2, lw, color=cb_palette[1])

    # get rectangle bounding boxes and calculate text positioning xs and ys
    bboxes = [rect.get_bbox() for rect in rects2]
    far_left = np.array([bb.xmin for bb in bboxes])

    heights = np.array([bb.height for bb in bboxes])
    ymaxes = np.array([bb.ymax for bb in bboxes])

    text_ys = ymaxes - 0.6 * heights
    xlims = ax.get_xlim()
    x_range = xlims[1] - xlims[0]
    spacer = x_range / 100
    text_xs = far_left - spacer

    # place labels directly next to left side of left-most bars
    for x, y, name in zip(text_xs, text_ys, I_nt.index):
        ax.text(x, y, name, va='center', ha='right',
                weight='bold', alpha=0.7)

    # remove yticks/grid and resize yaxis
    ax.set_yticks([])
    ax.yaxis.grid(False)

    # ax.set_ylim(0, max(yticks) + max(heights))
    ax.set_xlim(xlims[0] + 2 * spacer, xlims[1] - 2 * spacer)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    ax.set_ylim(ylims[0] - 1, max(ymaxes) + 3)
    ylims = ax.get_ylim()

    # Add top-level labels for "Transfer" and "Non-Transfer".
    # These specify which side of the graph is which.
    point0 = rects2[0].get_bbox().xmax
    x1 = point0 - 3 * spacer
    x2 = point0 + 3 * spacer
    ax.text(x1, ylims[1] - 2, 'Non-Transfer',
            fontsize='large', variant='small-caps', weight='bold',
            ha='right', va='bottom', color=cb_palette[1])
    ax.text(x2, ylims[1] - 2, 'Transfer',
            fontsize='large', variant='small-caps', weight='bold',
            ha='left', va='bottom', color=cb_palette[0])

    # add "%" to xticklabels
    xlabs = ax.get_xticks().tolist()
    ax.set_xticklabels(['%d%%' % abs(num) for num in xlabs])

    ax.set_title('Non-Transfer vs. Transfer Feature Importance Comparison')
    ax.set_xlabel('Percent of Absolute Deviation From Global Intercept')

    ax.figure.show()
    raw_input()
