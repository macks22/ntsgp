import os
import logging
from collections import OrderedDict

import luigi
import numpy as np
import pandas as pd

import test_params
from recpipe import UserCourseGradeLibFM, UsesTrainTestSplit


class UsesLibFM(UsesTrainTestSplit):
    """Base class for any class that uses libFM to produce results."""
    time = luigi.Parameter(
        default='',
        description='if empty; no time attributes, ' +
                    'cat = categorical encoding (TimeSVD), ' +
                    'bin = binary, one-hot encoding (BPTF)')
    iterations = luigi.IntParameter(
        default=150,
        description='number of iterations to use for learning')
    init_stdev = luigi.FloatParameter(
        default=0.5,
        description='initial std of Gaussian spread; higher can be faster')
    use_bias = luigi.BoolParameter(
        default=False,
        description='use global and per-feature bias terms if True')
    dim = luigi.IntParameter(
        default=7,
        description='dimensionality to use for matrix factorization')

    prefix = ''
    subtask_class = UserCourseGradeLibFM

    @property
    def libfm_arg_indicators(self):
        parts = []

        # time information part
        if self.time:
            parts.append('T%s' % self.time)

        # number of iterations part
        parts.append('i%d' % self.iterations)

        # initial standard deviation part (init_stdev)
        std = 's%s' % ''.join(str(self.init_stdev).split('.'))
        parts.append(std)

        # include dimensionality
        parts.append('dim%d' % self.dim)

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
    task = luigi.Parameter(
        default='next',
        description='prediction task; next = next-term, all = all-terms')
    base = 'outcomes'
    ext = 'tsv'

    base = 'predict'

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

    def output(self):
        parts = self.libfm_arg_indicators
        self.suffix = '-'.join(parts)
        base_fname = self.output_base_fname()
        subext = '{}.t%d'.format(self.__class__.__name__)
        error_ext = '{}.rmse'.format(self.__class__.__name__)

        outputs = {termnum: luigi.LocalTarget(base_fname % (subext % termnum))
                   for termnum in self.term_range}
        return {
            'error': luigi.LocalTarget(base_fname % error_ext),
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
        outfile = self.outputs()['predict'].path
        guide_file = inputs['guide']

        # run libFM to compute all predictions in one pass.
        results = self.libfm_predict(train, test, outfile)

        # Now match up predictions with labeled test examples.
        # First read the labeled test grades.
        with test_file.open() as f:
            test = pd.read_csv(f, sep=' ', usecols=[0], header=None)
            test = test.values[:,0]

        # The guide tells us which examples are from which term.
        with guide_file.open() as f:
            guide = pd.read_csv(f, index_col=0)

        # Calculate squared error for each term.
        error = OrderedDict()
        pos = 0
        for termnum in guide.index:
            last_rownum = guide.id[termnum] + 1
            predicted = results[pos: last_rownum]
            error[termnum] = abs(predicted - test) ** 2
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
            f.write('\t'.join(rmse_vals) + '\n')
            f.write('\t'.join(running_vals))


class SVD(RunLibFM):
    """Run libFM to emulate SVD."""
    use_bias = False
    time = ''

class BiasedSVD(SVD):
    """Run libFM to emulate biased SVD."""
    use_bias = True

class TimeSVD(SVD):
    """Run libFM to emulate TimeSVD."""
    time = 'cat'

class BiasedTimeSVD(TimeSVD):
    """Run libFM to emulate biased TimeSVD."""
    use_bias = True

class BPTF(RunLibFM):
    """Run libFM to emulate Bayesian Probabilistic Tensor Factorization."""
    use_bias = False
    time = 'bin'

class BiasedBPTF(BPTF):
    """Run libFM to emulate biased BPTF."""
    use_bias = True


class RunAllOnSplit(RunLibFM):
    """Run all available methods via libFM for a particular train/test split."""
    train_filters = luigi.Parameter(  # restate to make non-optional
        description='Specify how to split the train set from the test set.')
    time = ''     # disable parameter
    use_bias = '' # disable parameter
    subtask_class = RunLibFM

    def requires(self):
        return [
            SVD(**self.param_kwargs),
            BiasedSVD(**self.param_kwargs),
            TimeSVD(**self.param_kwargs),
            BiasedTimeSVD(**self.param_kwargs),
            BPTF(**self.param_kwargs),
            BiasedBPTF(**self.param_kwargs)
        ]

    def output(self):
        """ Each method returns a dictionary with the 'error' key containing a
        listing of term-by-term and overall RMSE, and the 'predict' key
        containing files with all grade predictions. We only want to pass on
        the error files, since the eventual goal is comparison between methods.
        """
        error_files = [in_dict['error'] for in_dict in self.input()]
        return [luigi.LocalTarget(f.path) for f in error_files]

    def extract_method_name(self, outfile):
        """We can pull these from the first prediction file. The method name is
        present before the last two extensions. For example: SVD.t9.pred.
        """
        subext = os.path.splitext(outfile)[0]
        base = os.path.splitext(base)[0]
        return os.path.splitext(base)[1].strip('.')

    @property
    def method_names(self):
        return [self.extract_method_name(in_dict['predict'].values()[0].path)
                for in_dict in self.input()]

    run = luigi.Task.run  # reset to default


class CompareMethods(RunAllOnSplit):
    """Aggregate results from all available methods on a particular split."""

    base = 'outcomes'
    ext = 'tsv'
    subtask_class = RunAllOnSplit

    def output(self):
        parts = self.libfm_arg_indicators
        self.suffix = '-'.join(parts)
        base_fname = self.output_base_fname()
        fname = base_fname % 'compare'
        return luigi.LocalTarget(fname)

    def requires(self):
        return self.subtask

    # TODO: LEFT OF HERE; FINISH UP WITH CLASS DEFINITION TO COMPARE RESULTS
    # ACROSS METHODS

    def read_results(self, f):
        content = f.read()
        rows = [l.split('\t') for l in content.split('\n')]
        rows = [[int(r[0]),float(r[1]),float(r[2])] for r in rows]
        return rows

    def run(self):
        results = []  # store results for all methods
        for input in self.input():
            with input.open() as f:
                rows = self.read_results(f)

            # add method name to each result for this method
            method_name = self.extract_method_name(input.path)
            for row in rows:
                row.insert(0, method_name)

            # keep the top 3 results for each method
            top = list(sorted(rows, key=lambda tup: tup[-1]))
            results += top[:self.topn]

        # now we have results from all methods, sort them
        top = list(sorted(results, key=lambda tup: tup[-1]))
        with self.output().open('w') as f:
            f.write('\t'.join(('method', 'dim', 'train', 'test')) + '\n')
            f.write('\n'.join(['\t'.join(map(str, row)) for row in top]))


class ResultsMarkdownTable(CompareMethods):
    """Produce markdown table of results comparison for a data split."""
    precision = luigi.IntParameter(
        default=5,
        description='number of decimal places to keep for error measurements')

    def requires(self):
        kwargs = self.param_kwargs.copy()
        del kwargs['precision']
        return CompareMethods(**kwargs)

    def output(self):
        outname = self.input().path
        base = os.path.splitext(outname)[0]
        return luigi.LocalTarget('%s.md' % base)

    def read_results(self, f):
        header = f.readline().strip().split('\t')
        content = f.read()
        rows = [l.split('\t') for l in content.split('\n')]
        fmt = '%.{}f'.format(self.precision)
        for row in rows:
            row[2] = fmt % float(row[2])
            row[3] = fmt % float(row[3])
        return header, rows

    def run(self):
        with self.input().open() as f:
            header, rows = self.read_results(f)

        # results are already sorted; we simply need to format them as a
        # markdown table; first find the column widths, leaving a bit of margin
        # space for readability
        widths = np.array([[len(item) for item in row]
                           for row in rows]).max(axis=0)
        margin = 4
        colwidths = widths + margin
        underlines = ['-' * width for width in widths]

        # next, justify the columns appropriately
        def format_row(row):
            return [row[0].ljust(colwidths[0])] + \
                   [row[i].rjust(colwidths[i]) for i in range(1, 4)]

        output = [format_row(header), format_row(underlines)]
        output += [format_row(row) for row in rows]

        # finally, write the table
        with self.output().open('w') as f:
            f.write('\n'.join([''.join(row) for row in output]))


class RunAll(luigi.Task):
    """Run all available methods on 0-4 and 0-7 train/test splits."""

    # The splits divide the data into these proportions (train | test)
    # ----------------------------------------------------------------
    # 0-1  (2009-2009): .282 | .718
    # 0-4  (2009-2010): .542 | .458
    # 0-7  (2009-2011): .758 | .242
    # 0-10 (2009-2012): .910 | .240

    splits = ["0-1", "0-4", "0-7", "0-10"]  # 4 splits
    backfills = [0, 1, 2, 3, 4, 5]  # 6 backfill settings

    @property
    def num_method_runs(self):
        """How many times libFM is run."""
        task = RunAllOnSplit(train_filters=self.splits[0])
        num_methods = len(task.deps())
        return num_methods * len(self.splits) * len(self.backfills)

    @property
    def num_iterations(self):
        """The total number of iterations libFM is run over all methods."""
        task = RunAllOnSplit(train_filters=self.splits[0])
        dim_range = task.dim_end - task.dim_start
        return task.iterations * dim_range * self.complexity

    # TODO: extend this to actually perform comparison between results
    def requires(self):
        for split in self.splits:
            for backfill in self.backfills:
                yield ResultsMarkdownTable(
                    train_filters=split,
                    backfill_cold_students=backfill,
                    backfill_cold_courses=backfill)


if __name__ == "__main__":
    luigi.run()
