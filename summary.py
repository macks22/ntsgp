"""
Print summary tables from results.

"""
import sys
import numpy as np


def summary(fname, concise=False):
    """The file being read has a table of the per-term error, and a second table
    with running mean of error. We want a table that just has the per-term error
    and the cumulative error. We can also discard the `rmse` column, which just
    specifies the difference between the tables.
    """
    with open(fname) as f:
        content = f.read()

    t1, t2 = content.split('\n\n')
    rows = [l.split() for l in t2.split('\n') if l]

    # extract mean RMSE for each method
    totals = [row[-1] for row in rows]
    totals[0] = "all"  # header
    totals[2] = ""     # remove count from last term

    # now remove second column from per-term table and insert total RMSEs
    rows = [l.split() for l in t1.split('\n') if l]
    for i in range(len(rows)):
        rows[i][1] = totals[i]  # replace junk with total RMSE


    # fix the "# test records" label
    rows[2][0] = "# predictions"
    rows[2].pop(2)

    # calculate total number of grades predicted
    total_count = sum(map(int, rows[2][2:]))
    rows[2][1] = str(total_count)

    # remove old underlining
    rows.pop(1)

    # EXPERIMENTAL
    if concise:
        for i in range(2, len(rows[0])):
            rows[0][i] = rows[0][i].replace('term', 't')

        for i in range(2, len(rows)):
            for j in range(1, len(rows[i])):
                rows[i][j] = rows[i][j][:5]

    # compute margin for justification formatting
    widths = np.array([[len(item) for item in row]
                       for row in rows]).max(axis=0)
    margin = 4
    colwidths = np.array(widths) + margin
    underlines = ['-' * width for width in widths]
    rows.insert(1, underlines)

    # next, justify the columns appropriately
    def format_row(row):
        return [row[i].ljust(colwidths[i]) for i in range(0, 2)] + \
               [row[i].rjust(colwidths[i]) for i in range(2, len(row))]

    lines = [''.join(format_row(row)) for row in rows]
    return '\n'.join(lines)


if __name__ == "__main__":
    fname = sys.argv[1]
    print summary(fname)
