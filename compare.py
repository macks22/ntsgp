import os

import luigi
import numpy as np
import pandas as pd
import ujson as json

from libfm import RunLibFM
import summary


class RunFeatureVariants(RunLibFM):
    """Run with various feature combinations for comparison."""
    cfg = luigi.Parameter(
        description='json config file to read feature combos from')

    def read_config(self, cfg_file):
        """Read dictionary of feature combos to run from JSON config file."""
        with open(cfg_file) as f:
            config = json.load(f)
            return config.values()

    @property
    def basekwargs(self):
        kwargs = self.param_kwargs.copy()
        # Disable all features to start.
        del kwargs['cfg']
        for featname in self.possible_features:
            kwargs[featname] = False
        return kwargs

    def requires(self):
        tasks = []  # store tasks to run; will be returned
        kwargs = self.basekwargs
        feature_combos = self.read_config(self.cfg)

        # now set up all tasks for requested feature combos
        for feats in feature_combos:  # list of lists
            params = kwargs.copy()
            for featname in feats:
                params[featname] = True
            tasks.append(RunLibFM(**params))

        return tasks

    def output(self):
        """ Each method returns a dictionary with the 'error' key containing a
        listing of term-by-term and overall RMSE, and the 'predict' key
        containing files with all grade predictions. We only want to pass on
        the error files, since the eventual goal is feature info comparison.
        """
        error_files = [in_dict['error'] for in_dict in self.input()]
        return [luigi.LocalTarget(f.path) for f in error_files]

    def extract_feat_abbrev(self, path):
        parts = path.split('-')
        return parts[1] if parts[1] != 'nocs' else parts[2]

    def feat_combos(self):
        return [self.extract_feat_abbrev(f.path) for f in self.output()]

    run = luigi.Task.run  # set to default


class CompareFeatureVariants(RunFeatureVariants):
    """Compare results from including various features."""
    comparison_name = luigi.Parameter(
        description='memorable name for comparison')
    base = 'outcomes'
    ext = 'tsv'
    subtask_class = RunFeatureVariants

    def output(self):
        base_fname = self.base_outfile_name
        fname = base_fname % ('compare-%s' % self.comparison_name)
        return luigi.LocalTarget(fname)

    def requires(self):
        return self.subtask

    def read_results(self, f):
        """Each file has a header, with the term numbers, a row of RMSE scores
        per term, and then a final row of running average RMSE.
        """
        return [l.split('\t') for l in f.read().split('\n')]

    def feat_combos(self):
        return [self.extract_feat_abbrev(f.path) for f in self.input()]

    def run(self):
        results = {}
        for f in self.input():
            name = self.extract_feat_abbrev(f.path)
            with f.open() as f:
                header, counts, perterm, running = self.read_results(f)
                results[name] = [perterm, running]

        # now we have results from all methods, sort them by total rmse
        records = results.items()
        total_rmse = lambda pair: pair[1][1][-1]
        records.sort(key=total_rmse)
        head = '\t'.join(['method', 'rmse'] + header)
        with self.output().open('w') as f:
            f.write('%s\n' % head)
            f.write('%s\n' % '\t'.join(['', ''] + counts))
            for name, (perterm, _) in records:
                f.write('%s\n' % '\t'.join([name, 'per-term'] + perterm))

            f.write('\n')
            for name, (_, running) in records:
                f.write('%s\n' % '\t'.join([name, 'running'] + running))


class CompareMdTable(CompareFeatureVariants):
    """Produce markdown table of results comparison for a data split."""

    subtask_class = CompareFeatureVariants

    def output(self):
        outname = self.input().path
        base = os.path.splitext(outname)[0]
        return luigi.LocalTarget('%s.md' % base)

    def read_results(self, f):
        header = f.readline().strip().split('\t')
        counts = ['# test records', ''] + f.readline().strip().split('\t')
        content = f.read()
        rows = [l.split('\t') for l in content.split('\n')]
        return header, counts, rows

    def run(self):
        with self.input().open() as f:
            header, counts, rows = self.read_results(f)

        # results are already sorted; we simply need to format them as a
        # markdown table; first find the column widths, leaving a bit of margin
        # space for readability
        widths = np.array([[len(item) for item in row]
                           for row in rows]).max(axis=0)
        margin = 4
        colwidths = np.array(widths) + margin
        underlines = ['-' * width for width in widths]

        # next, justify the columns appropriately
        def format_row(row):
            return [row[i].ljust(colwidths[i]) for i in range(0, 2)] + \
                   [row[i].rjust(colwidths[i]) for i in range(2, len(row))]

        table1 = [format_row(header), format_row(underlines), format_row(counts)]
        table2 = table1[:]
        for row in rows:
            if row and row[-1]:
                if row[1] == 'per-term':
                    table1.append(format_row(row))
                else:
                    table2.append(format_row(row))

        # finally, write the tables
        with self.output().open('w') as f:
            f.write('\n'.join([''.join(row) for row in table1]) + '\n\n')
            f.write('\n'.join([''.join(row) for row in table2]))


class CompareSummary(CompareMdTable):
    subtask_class = CompareMdTable

    def output(self):
        outname = self.input().path
        base = os.path.splitext(outname)[0]
        return luigi.LocalTarget('%s-summ.md' % base)

    def run(self):
        with self.input().open() as f:
            report = summary.summary(f)

        with self.output().open('w') as f:
            f.write(report)


if __name__ == "__main__":
    luigi.run()
