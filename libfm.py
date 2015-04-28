import os
import logging
import itertools as it
from collections import OrderedDict

import luigi
import numpy as np
import pandas as pd

import test_params
from util import *
import summary
from recpipe import UserCourseGradeLibFM, UsesFeatures


class UnexpectedNaN(Exception):
    """Raise when np.nan values are found unexpectedly."""
    pass


class UsesLibFM(UsesFeatures):
    """Base class for any class that uses libFM to produce results."""
    task = luigi.Parameter(
        default='next',
        description='prediction task; next = next-term, all = all-terms')
    iterations = luigi.IntParameter(
        default=150,
        description='number of iterations to use for learning')
    init_stdev = luigi.FloatParameter(
        default=0.5,
        description='initial std of Gaussian spread; higher can be faster')
    use_bias = luigi.BoolParameter(
        default=False,
        description='use per-feature bias terms if True')
    dim = luigi.IntParameter(
        default=7,
        description='dimensionality to use for matrix factorization')

    suffix = ''
    prefix = ''
    subtask_class = UserCourseGradeLibFM

    @property
    def libfm_arg_indicators(self):
        parts = []

        # Add feature names (abbreviated) to path name.
        if self.features:
            parts.append(abbrev_names(self.features))

        # number of iterations part
        parts.append('i%d' % self.iterations)

        # initial standard deviation part (init_stdev)
        std = 's%s' % ''.join(str(self.init_stdev).split('.'))
        parts.append(std)

        # include dimensionality
        parts.append('d%d' % self.dim)

        # bias terms part
        if self.use_bias:
            parts.append('b')

        return parts

    def filter_kwargs(self, task):
        task_params = [tup[0] for tup in task.get_params()]
        return {k:v for k, v in self.param_kwargs.items() if k in task_params}

    @property
    def subtask(self):
        try:
            return self._subtask
        except:
            task = self.subtask_class
            self._subtask = task(**self.filter_kwargs(task))
            return self._subtask

    def requires(self):
        return self.subtask

    def output(self):
        """Generic output function for any task that runs libFM."""
        parts = self.libfm_arg_indicators
        self.suffix = '-'.join(parts)
        base_fname = self.output_base_fname()
        fname = base_fname % self.__class__.__name__
        return luigi.LocalTarget(fname)

    @property
    def common_kwargs(self):
        return {
            'dim': self.dim,
            'std': self.init_stdev,
            'bias': self.use_bias,
            'iter': self.iterations
        }

    @property
    def libfm_command(self):
        def show_libfm_command(train_fname, test_fname, outfile=''):
            return ' '.join(test_params.compose_libfm_args(
                train_fname, test_fname, outfile=outfile, **self.common_kwargs))
        return show_libfm_command

    @property
    def run_libfm(self):
        def run_libfm(train_fname, test_fname, outfile=''):
            return test_params.run_libfm(
                train_fname, test_fname, outfile=outfile, **self.common_kwargs)
        return run_libfm

    @property
    def libfm_predict(self):
        def libfm_predict(train_fname, test_fname, outfile):
            return test_params.libfm_predict(
                train_fname, test_fname, outfile=outfile, **self.common_kwargs)
        return libfm_predict


class RunLibFM(UsesLibFM):
    """General-purpose wrapper that spawns a subprocess to run libFM."""
    remove_cold_start = luigi.IntParameter(
        default=1,
        description="remove all cold-start records from test set")
    base = 'predict'
    ext = 'tsv'

    @property
    def guide(self):
        if self.task == 'all':
            with self.input()['guide'].open() as f:
                return pd.read_csv(f, index_col=0)
        else:
            return None

    @property
    def term_range(self):
        if self.task ==  'all':
            return self.guide.index
        else:
            return self.subtask.term_range

    @property
    def base_outfile_name(self):
        parts = self.libfm_arg_indicators
        parts.append('aterm' if self.task == 'all' else 'nterm')
        self.suffix = '-'.join(parts)
        return self.output_base_fname()

    def output(self):
        base_fname = self.base_outfile_name
        subext = '{}.t%d'.format(self.__class__.__name__)
        error_ext = '{}.rmse'.format(self.__class__.__name__)
        error_fname = base_fname % error_ext

        if self.task == 'next':
            outputs = \
                {termnum: luigi.LocalTarget(base_fname % (subext % termnum))
                 for termnum in self.term_range}
        else:
            outputs = luigi.LocalTarget(base_fname % self.__class__.__name__)

        return {
            'error': luigi.LocalTarget(error_fname),
            'predict': outputs
        }

    def next_term_prediction(self):
        """Run libFM over per-term splits for next-term prediction."""
        inputs = self.input()
        outputs = self.output()
        error = OrderedDict()
        logging.info('%s: next-term prediction for %d terms' % (
            self.__class__.__name__, len(inputs)))

        # Run libFM to predict grades for each term
        for termnum in inputs:
            logging.info('predicting grades for term %d' % termnum)
            train = inputs[termnum]['train'].path
            test_file = inputs[termnum]['test']
            test = test_file.path
            outfile = outputs['predict'][termnum].path
            predicted = self.libfm_predict(train, test, outfile)

            # Now calculate absolute deviation of predictions from actuals
            with test_file.open() as f:
                test = pd.read_csv(f, sep=' ', usecols=[0], header=None)
                test = test.values[:,0]

            error[termnum] = abs(predicted - test) ** 2

        return error

    def all_term_prediction(self):
        """Run libFM on the single train/test split for all-term prediction."""
        inputs = self.input()
        train = inputs['train'].path
        test_file = inputs['test']
        test = test_file.path
        outfile = self.output()['predict'].path
        guide_file = inputs['guide']

        # run libFM to compute all predictions in one pass.
        results = self.libfm_predict(train, test, outfile)
        nan_mask = np.isnan(results)
        if nan_mask.any():
            raise UnexpectedNaN(
                "%d np.nan values in libFM predictions." % nan_mask.sum())

        # Now match up predictions with labeled test examples.
        # First read the labeled test grades.
        with test_file.open() as f:
            test = pd.read_csv(f, sep=' ', usecols=[0], header=None)
            test = test.values[:,0]

        nan_mask = np.isnan(test)
        if nan_mask.any():
            raise UnexpectedNaN(
                "%d np.nan values in test values." % nan_mask.sum())

        # The guide tells us which examples are from which term.
        with guide_file.open() as f:
            guide = pd.read_csv(f, index_col=0)

        # Calculate squared error for each term.
        error = OrderedDict()
        pos = 0
        for termnum in guide.index:
            last_rownum = guide.rownum[termnum] + 1
            predicted = results[pos: last_rownum]
            testvals = test[pos: last_rownum]
            error[termnum] = abs(predicted - testvals) ** 2
            pos = last_rownum

        return error

    def run(self):
        # Calculate squred error per term
        if self.task == 'all':
            sqerror = self.all_term_prediction()
        else:
            sqerror = self.next_term_prediction()

        # compute rmse by term and over all terms
        err_arrays = sqerror.values()
        counts = np.array([len(errvals) for errvals in err_arrays])
        err_sums = np.array([errvals.sum() for errvals in err_arrays])
        rmse_vals = np.sqrt(err_sums / counts)

        # compute running mean
        running_mean = [rmse_vals[0]]
        total_cnt = counts[0]
        for i in range(1, len(rmse_vals)):
            newcount = total_cnt + counts[i]
            running_mean.append(
                ((running_mean[i-1] * total_cnt + rmse_vals[i] * counts[i]) /
                 newcount))
            total_cnt = newcount

        # write all error calculations
        rmse_vals = ['%.5f' % val for val in rmse_vals]
        running_vals = ['%.5f' % val for val in running_mean]
        header = ['term%d' % tnum for tnum in sqerror]
        with self.output()['error'].open('w') as f:
            f.write('\t'.join(header) + '\n')
            f.write('\t'.join(map(str, counts)) + '\n')
            f.write('\t'.join(rmse_vals) + '\n')
            f.write('\t'.join(running_vals))


class RunFeatureCombinations(RunLibFM):
    """Run with various feature combinations; allow |F| choose nfeats."""
    nfeats = luigi.IntParameter(
        default=1, description='how many params to use in combinations')

    @property
    def basekwargs(self):
        kwargs = self.param_kwargs.copy()
        del kwargs['nfeats']  # remove params unique to this class
        # Disable all features to start.
        for featname in self.possible_features:
            kwargs[featname] = False
        return kwargs

    def requires(self):
        tasks = []  # store tasks to run; will be returned
        kwargs = self.basekwargs

        # now get all possible combinations
        for attrs in it.combinations(self.possible_features, self.nfeats):
            # enable parameters from this combination
            params = kwargs.copy()
            for attr in attrs:
                params[attr] = True
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


class CompareFeatures(RunFeatureCombinations):
    """Compare results from including various features."""
    base = 'outcomes'
    ext = 'tsv'
    subtask_class = RunFeatureCombinations

    def output(self):
        base_fname = self.base_outfile_name
        fname = base_fname % ('compare-nf%d' % self.nfeats)
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


class ResultsMarkdownTable(CompareFeatures):
    """Produce markdown table of results comparison for a data split."""

    subtask_class = CompareFeatures

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


class ResultsSummary(ResultsMarkdownTable):
    subtask_class = ResultsMarkdownTable

    def output(self):
        outname = self.input().path
        base = os.path.splitext(outname)[0]
        return luigi.LocalTarget('%s-summ.md' % base)

    def run(self):
        with self.input().open() as f:
            report = summary.summary(f)

        with self.output().open('w') as f:
            f.write(report)


# TODO: update for feature comparison.
class RunAll(luigi.Task):
    """Run all available methods on 0-4 and 0-7 train/test splits."""
    iterations = luigi.IntParameter(
        default=150,
        description='number of iterations to use for learning')
    init_stdev = luigi.FloatParameter(
        default=0.5,
        description='initial std of Gaussian spread; higher can be faster')
    discard_nongrade = luigi.Parameter(
        default=True,
        description='drop W/S/NC grades from training data if True')
    remove_cold_start = luigi.IntParameter(
        default=1,
        description="remove all cold-start records from test set")

    # The splits divide the data into these proportions (train | test)
    # ----------------------------------------------------------------
    # 0-1  (2009-2009): .282 | .718
    # 0-4  (2009-2010): .542 | .458
    # 0-7  (2009-2011): .758 | .242
    # 0-10 (2009-2012): .910 | .240

    splits = ["0-1", "0-4", "0-7", "0-10"]  # 4 splits
    tasks = ["all", "next"]  # 2 prediction tasks

    # Backfilling may be an unfair way to mix the data -- definitely for
    # courses, and likely for students as well.
    # backfills = [0, 1, 2, 3, 4, 5]  # 6 backfill settings

    @property
    def num_method_runs(self):
        """How many times libFM is run."""
        task = RunAllOnSplit(train_filters=self.splits[0], **self.param_kwargs)
        num_methods = len(task.deps())
        return num_methods * len(self.splits) * len(self.tasks)

    @property
    def num_iterations(self):
        """The total number of iterations libFM is run over all methods."""
        task = RunAllOnSplit(train_filters=self.splits[0], **self.param_kwargs)
        return task.iterations * self.num_method_runs

    # TODO: extend this to actually perform comparison between results
    def requires(self):
        for split in self.splits:
            for task in self.tasks:
                yield ResultsMarkdownTable(
                    train_filters=split, task=task, **self.param_kwargs)


if __name__ == "__main__":
    luigi.run()
