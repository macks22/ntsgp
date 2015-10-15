import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

from imps import read_frames_and_counts

plt.style.use('ggplot')
cb_palette = sns.color_palette('colorblind')


def plot_smts(tnums, dfs, colname='Importance', tmap=None):
    """Plot a time-series of small-multiples given an iterable of data
    frames. Each individual data frame is assumed to contain feature importance
    metrics in `colname`. The index should be the feature names.
    """
    # Sort data frames by ascending term number.
    ordered = [(tnum, df) for tnum, df in sorted(zip(tnums, dfs))]
    tnums = [tnum for tnum, df in ordered]
    dfs = [df for tnum, df in ordered]

    # Lay out subplot grid.
    fig = plt.figure()
    ntime_units = len(dfs)
    gs = gridspec.GridSpec(1, ntime_units)
    gs.update(wspace=0.025)  # remove extra space between axes
    axes = [fig.add_subplot(gs[i]) for i in range(ntime_units)]

    df0 = dfs[0]
    n = df0.shape[0]
    idx = np.arange(n)
    lw = 2

    # Figure out how we should scale the x-axis.
    # We need enough space for the largest feature importance encountered.
    # So we look through all data frames and get the max importance.
    imp_max = pd.concat(dfs)[colname].max()

    # Sort the first data frame in order of highest to lowest feature
    # importance, then use this ordering to sort all subsequent frames.
    ordered_idx = df0.sort(colname, ascending=True).index
    for i in range(ntime_units):
        dfs[i].index = pd.Categorical(dfs[i].index, ordered_idx)
        dfs[i] = dfs[i].sort_index()

        df = dfs[i]
        ax = axes[i]
        termnum = tnums[i]

        # First make a bar plot in order to get the general layout
        rects = ax.barh(idx, df[colname], lw, color='black')
        bboxes = [rect.get_bbox() for rect in rects]
        heights = np.array([bb.height for bb in bboxes])
        ymaxes = np.array([bb.ymax for bb in bboxes])
        ymids = ymaxes - 0.5 * heights
        xmaxes = np.array([bb.xmax for bb in bboxes])
        xmins = np.array([0.0 for bb in bboxes])

        # Create the points that represent the start/end of each line segment.
        p1s = zip(xmins, ymids)
        p2s = zip(xmaxes, ymids)
        pt_to_pt = zip(p1s, p2s)

        # Next remove the rectangles and draw black lines in their place.

        # Line2D takes a list of x values and a list of y values, not 2 points
        # So we convert our points an x-list and a y-list.
        xs = np.zeros((n, 2))
        ys = np.zeros((n, 2))
        for j, line in enumerate(pt_to_pt):
            xs[j], ys[j] = zip(*line)
            rects[j].remove()
            ax.add_line(plt.Line2D(xs[j], ys[j], lw=lw, color='black'))

        # Resize graph dimensions; remove ygrid.
        ax.yaxis.grid(False)
        ax.set_ylim(0, max(ymaxes) + 1)
        ax.set_xlim(0, imp_max + 0.01)
        xlims = ax.get_xlim()
        x_range = xlims[1] - xlims[0]

        # Connect a line along the y-axis to form a base
        # for the importance lines.
        ax.add_line(plt.Line2D((0.003, 0.003), (min(ymids), max(ymids)),
                    lw=1, color='black'))

        # Label x-axis with semester name.
        lab_x = xlims[0] + 0.5 * x_range
        lab_y = min(ymids)
        ax.text(lab_x, lab_y * 0.25,
                tmap.ix[termnum]['short'], fontsize=7, ha='center', va='top')


    # Add y-axis tick labels to left-most subplot and remove other plot labels.
    axes[0].set_yticks(ymids)
    axes[0].set_yticklabels(ordered_idx, fontsize=7)
    for i in range(1, ntime_units):
        axes[i].set_yticks([])

    # Keep only highest x-axis label
    for i in range(ntime_units):
        ax = axes[i]
        xlabs = ax.get_xticks()[::2]  # reduce to half for sparser grid
        ax.set_xticks(xlabs)
        ax.set_axisbelow(False)  # grid lines slice up lines
        ax.set_axis_bgcolor('white')  # remove ggplot gray background
        xglines = plt.getp(ax, 'xgridlines')
        plt.setp(xglines, 'linewidth', 1.5)  # increase gridline thickness
        ax.set_xticklabels([])

    # Set figure title and xlabel
    fig.suptitle('Evolution of Feature Importance Over Semesters')
    fig.text(0.5, 0.05,
             'Each line chunk represents 5% of the Per-Semester '
             'Normalized Importance',
             ha='center', va='top')

    return fig


if __name__ == "__main__":
    dirname = sys.argv[1]
    tnums, dfs, counts = read_frames_and_counts(dirname, 'imp.csv')
    tmap = pd.read_csv(sys.argv[2], index_col=0)

    non_summer_terms = [tnum for tnum in tnums
                       if not 'Su' in tmap.ix[tnum]['name']]
    non_summers_dfs = [dfs[tnums.index(tnum)] for tnum in non_summer_terms]

    # fig = plot_smts(tnums, dfs, 'Importance', tmap)
    fig = plot_smts(non_summer_terms, non_summers_dfs, 'Importance', tmap)
    fig.show()
    fname = raw_input("Enter name for figure save (with ext): ")
    fig.savefig(fname)
