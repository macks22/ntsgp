import os
from collections import OrderedDict

import luigi
import pandas as pd
import numpy as np

from recpipe import UserCourseGradeLibFM, UsesLibFM, UsesTrainTestSplit


class UCGLibFMByTerm(UserCourseGradeLibFM):
    """Write UCG data for term-by-term evaluation."""

    base = 'tmp'

    @property
    def train(self):
        try:
            return self._train
        except:
            self._train, self._test = self.prep_data()
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self):
        try:
            return self._test
        except:
            self._train, self._test = self.prep_data()
        return self._test

    @test.setter
    def test(self, test):
        self._test = test

    @property
    def term_range(self):
        # Due to the backfilling, we must rely on the train filters to get the
        # last term in the training data.
        start = max([f.cohort_end for f in self.filters])
        end = int(self.test.cohort.max())
        return range(start + 1, end + 1)

    def output(self):
        """The filenames are written in such a way that the train/test
        sub-extensions have the number of the term the split should be used to
        predict. So if the sub-extensions are train5/test5, this split should be
        used to predict the grades for termnum 5.

        """
        fname = self.output_base_fname()
        outputs = {}
        for termnum in self.term_range:  # don't write files for last num
            train = '%s%d' % ('train', termnum)
            test = '%s%d' % ('test', termnum)
            outputs[termnum] = {
                'train': luigi.LocalTarget(fname % train),
                'test': luigi.LocalTarget(fname % test)
            }
        return outputs

    def transfer_term(self, termnum):
        """Move data for the given term from the test set to the train set."""
        tomove_mask = self.test.termnum == termnum
        self.train = pd.concat((self.train, self.test[tomove_mask]))
        self.test = self.test[~tomove_mask]

    def run(self):
        """Now the idea here is that the task at hand is predicting the grades
        for the next term. At the time we do this, we have the data available
        for all previous terms. Hence, we would like to output the initial
        train/test specified by the user, as usual. But then we also want to
        output another train/test set for each subsequent term.

        So if the user specifies 0-7, the first train set will have 0-7, and
        the test set will have 8-14. Then we'll output another 6 splits. For
        each new split, we find the max term number currently in the test set
        and transfer the term after that one from the test set to the train
        set. We continue until the test set includes only the last term.

        """
        train, test = self.train, self.test

        # Determine what to do with time
        write_libfm_data = self.multiplex_time(train, test)

        # write all data files
        outputs = self.output()
        for termnum in self.term_range:  # includes (end term + 1)
            test = self.test[self.test.termnum == termnum]
            for name, dataset in zip(['train', 'test'], [self.train, test]):
                with outputs[termnum][name].open('w') as f:
                    write_libfm_data(f, dataset)
            self.transfer_term(termnum)  # modify train/test sets in place
            # intentionally skip writing the last time this is run

        # TODO: this code converts the same records to libFM format multiple
        # times. Each subsequent train set contains records in the last train
        # set. These could be cached to avoid all of the string format
        # conversion overhead.


class RunLibFMByTerm(UsesLibFM, UsesTrainTestSplit):
    """Run libFM on prediction task, evaluating term by term."""

    # note that only dim_start is used; fixed dimension

    base = 'outcomes'
    ext = 'pred'

    @property
    def task(self):
        if getattr(self, '_task', ''):
            return self._task

        task_params = [tup[0] for tup in UCGLibFMByTerm.get_params()]
        kwargs = self.param_kwargs
        # kwargs['time'] = getattr(self, 'time', '')
        # kwargs['use_bias'] = getattr(self, 'use_bias', False)
        params = {k:v for k, v in kwargs.items() if k in task_params}
        self._task = UCGLibFMByTerm(**params)
        return self._task

    def requires(self):
        return self.task

    def output(self):
        """The outputs will actually be predictions from libFM."""
        parts = self.libfm_arg_indicators
        self.suffix = '-'.join(parts)
        base_fname = self.output_base_fname()
        subext = '{}.t%d'.format(self.__class__.__name__)

        return {termnum: luigi.LocalTarget(base_fname % (subext % termnum))
                for termnum in self.task.term_range}

    def run(self):
        """No need to write anything; simply pass output filenames to libFM."""
        inputs = self.input()
        outputs = self.output()
        for termnum in inputs:
            train = inputs[termnum]['train'].path
            test = inputs[termnum]['test'].path
            outfile = outputs[termnum].path
            self.run_libfm(train, test, outfile)


class EvalResultsByTerm(RunLibFMByTerm):
    """Calculate RMSE for each term and over all terms."""

    ext = 'tsv'

    def requires(self):
        # TODO: error coming from here
        return {
            'in': self.task,  # comes from inherited __init__
            'predict': RunLibFMByTerm(**self.param_kwargs)
        }

    def output(self):
        parts = self.libfm_arg_indicators
        self.suffix = '-'.join(parts)
        base_fname = self.output_base_fname()
        fname = base_fname % self.__class__.__name__
        return luigi.LocalTarget(fname)

    def rmse(self, error):
        mse = ((error ** 2).sum() / len(error)).values[0]
        return np.sqrt(mse)

    def run(self):
        inputs = self.input()
        error_by_term = OrderedDict()
        for termnum in inputs['in']:
            testfile = inputs['in'][termnum]['test']
            predict_file = inputs['predict'][termnum]

            with testfile.open() as f:
                test = pd.read_csv(f, sep=' ', usecols=[0], header=None)

            with predict_file.open() as f:
                predicted = pd.read_csv(f, header=None)

            # calculate absolute deviation from actual prediction
            error_by_term[termnum] = abs(test - predicted)

        # compute rmse by term and over all terms
        all_error = pd.concat(error_by_term.values())
        total_rmse = self.rmse(all_error)
        term_rmses = [self.rmse(error) for error in error_by_term.values()]
        rmse_vals = ['%.5f' % val for val in [total_rmse] + term_rmses]

        # write all error calculations
        header = ['all'] + ['term%d' % tnum for tnum in error_by_term.keys()]
        with self.output().open('w') as f:
            f.write('\t'.join(header) + '\n')
            f.write('\t'.join(rmse_vals))


class SVDByTerm(EvalResultsByTerm):
    """Run libFm to emulate SVD for term-by-term prediction."""
    use_bias = False
    time = ''

class BiasedSVDByTerm(SVDByTerm):
    """Run libFM to emulate biased SVD for term-by-term prediction."""
    use_bias = True

class TimeSVDByTerm(SVDByTerm):
    """Run libFM to emulate TimeSVD for term-by-term prediction."""
    time = 'cat'

class BiasedTimeSVDByTerm(TimeSVDByTerm):
    """Run libFM to emulate biased TimeSVD for term-by-term prediction."""
    use_bias = True

class BPTFByTerm(EvalResultsByTerm):
    """Run libFM to emulate BPTF for term-by-term prediction."""
    use_bias = False
    time = 'bin'

class BiasedBPTFByTerm(BPTFByTerm):
    """Run libFM to emulate biased BPTF for term-by-term prediction."""
    use_bias = True


class RunAllOnSplitByTerm(EvalResultsByTerm):
    """Run all available methods via libFM for a particular train/test split."""
    train_filters = luigi.Parameter(  # restate to make non-optional
        description='Specify how to split the train set from the test set.')
    time = ''     # disable parameter
    use_bias = '' # disable parameter

    def requires(self):
        kwargs = self.param_kwargs
        # if 'time' in kwargs:
        #     del kwargs['time']
        # if 'use_bias' in kwargs:
        #     del kwargs['use_bias']
        return [
            SVDByTerm(**kwargs),
            BiasedSVDByTerm(**kwargs),
            TimeSVDByTerm(**kwargs),
            BiasedTimeSVDByTerm(**kwargs),
            BPTFByTerm(**kwargs),
            BiasedBPTFByTerm(**kwargs)
        ]

    def output(self):
        return [luigi.LocalTarget(f.path) for f in self.input()]

    def extract_method_name(self, outfile):
        base = os.path.splitext(outfile)[0]
        method = os.path.splitext(base)[1].strip('.')
        return method.replace('ByTerm', '')

    @property
    def method_names(self):
        return [self.extract_method_name(f.path) for f in self.input()]

    run = luigi.Task.run  # reset to default


class CompareMethodsByTerm(RunAllOnSplitByTerm):
    """Aggregate results from all available methods on a particular split."""

    base = 'outcomes'
    ext = 'tsv'
    suffix = ''

    def output(self):
        parts = self.libfm_arg_indicators
        self.suffix = '-'.join(parts)
        base_fname = self.output_base_fname()
        fname = base_fname % 'byterm.dim%d' % self.dim_start
        return luigi.LocalTarget(fname)

    def requires(self):
        return RunAllOnSplitByTerm(train_filters=self.train_filters)

    def read_results(self, f):
        headers = f.readline()
        rmse_vals = f.readline().strip().split('\t')
        return map(float, rmse_vals)

    @property
    def header(self):
        with self.input()[0].open() as f:
            headers = f.readline().strip().split('\t')
        return ['method'] + headers

    def run(self):
        results = []  # store results for all methods
        for input in self.input():
            with input.open() as f:
                errors = self.read_results(f)

            # add method name to each result for this method
            method_name = self.extract_method_name(input.path)
            errors.insert(0, method_name)
            results.append(errors)

        # now we have results from all methods, sort them
        top = list(sorted(results, key=lambda tup: tup[1]))
        with self.output().open('w') as f:
            f.write('\t'.join(self.header) + '\n')
            f.write('\n'.join(['\t'.join(map(str, row)) for row in top]))


if __name__ == "__main__":
    luigi.run()
