import os
import csv
import subprocess as sub

import luigi
import numpy as np
import pandas as pd

import test_params
from writer import write_libfm, write_triples
from preprocess import preprocess, CVALS, RVALS
from util import *


class BasicLuigiTask(luigi.Task):
    """Uses class name as output file."""
    ext = 'csv'
    def output(self):
        fname = fname_from_cname(self.__class__.__name__)
        return luigi.LocalTarget(os.path.join(
            'data', '%s.%s' % (fname, self.ext)))


class PreprocessedData(BasicLuigiTask):
    """Loads in preprocessed course data."""
    def run(self):
        with self.output().open('w') as f:
            self.cvals, self.rvals = preprocess(f)


class TrainTestFilter(object):
    """Wrapper class to filter data to train/test sets using cohort/term."""
    term_max = 14  # some number greater than max term id

    def __init__(self, filt):
        if ':' in filt:
            cohort, term = filt.split(':')
            self.cohort_start, self.cohort_end = self._split(cohort)
            self.term_start, self.term_end = self._split(term)
        else:
            self.cohort_start, self.cohort_end = map(int, filt.split('-'))
            self.term_start, self.term_end = (0, self.term_max)

    def _split(self, config):
        if '-' in config:
            return map(int, config.split('-'))
        else:
            return (int(config), self.term_max)

    def __str__(self):
        return '%d_%dT%d_%d' % (
            self.cohort_start, self.cohort_end, self.term_start, self.term_end)

    def mask(self, data):
        return ((data['cohort'] >= self.cohort_start) &
                (data['cohort'] <= self.cohort_end) &
                (data['termnum'] >= self.term_start) &
                (data['termnum'] <= self.term_end))

    def train(self, data):
        return data[self.mask(data)]

    def test(self, data):
        return data[~self.mask(data)]


class UsesTrainTestSplit(luigi.Task):
    """Base task for train/test split args and filters init."""
    train_filters = luigi.Parameter(
        default='0-14',
        description='Specify how to split the train set from the test set.')
    remove_cold_start = luigi.IntParameter(
        default=1,
        description="remove all cold-start records from test set")

    base = 'data'  # directory to write files to
    ext = 'tsv'    # final file extension for output files
    prefix = 'ucg' # prefix for all output files
    suffix = ''    # class-specific suffix that goes before ext on output names

    possible_features = CVALS + RVALS
    for featname in possible_features:
        locals()[featname] = luigi.BoolParameter(default=False)

    # Required features.
    locals()['sid'] = True
    locals()['cid'] = True
    locals()['grdpts'] = True

    @property
    def cvals(self):
        return [cval for cval in CVALS if getattr(self, cval)]

    @property
    def rvals(self):
        return [rval for rval in RVALS if getattr(self, rval)]

    @property
    def features(self):
        """List all features that will be used."""
        return self.cvals + self.rvals

    @property
    def featnames(self):
        return abbrev_names(self.features)

    @property
    def filters(self):
        return [TrainTestFilter(filt) for filt in self.train_filters.split()]

    def output_base_fname(self):
        parts = [self.prefix] if self.prefix else []

        # parameter suffix part
        param_suffix = '-'.join([str(filt) for filt in self.filters])
        if param_suffix:
            parts.append(param_suffix)

        # indicate whether cold-start backfilling was done for students/courses
        if self.remove_cold_start:
            parts.append('nocs')

        # include optional class-specific suffix
        if self.suffix:
            parts.append(self.suffix)

        if self.featnames:
            parts.append(self.featnames)

        fbase = os.path.join(self.base, '-'.join(parts))
        return '{}.%s.{}'.format(fbase, self.ext)


class DataSplitterBaseTask(UsesTrainTestSplit):
    """Functionality to split train/test data, no run method."""
    data_source = PreprocessedData()

    def requires(self):
        return self.data_source

    def split_data(self):
        with self.input().open() as f:
            data = pd.read_csv(f)

        # sort data by term number, then by student id
        data = data.sort(['termnum', 'sid'])

        # now do train/test split; drop duplicates in case filters overlap
        train = pd.concat([f.train(data) for f in self.filters]).drop_duplicates()
        test = pd.concat([f.test(data) for f in self.filters]).drop_duplicates()

        # sometimes cohorts have nan values, and other times students from later
        # cohorts take courses before they've officially enrolled.
        start = max([f.cohort_end for f in self.filters])
        oddball_mask = test.termnum <= start
        train = pd.concat((train, test[oddball_mask]))
        test = test[~oddball_mask]

        return (train, test)


class LibFMAllTermInput(DataSplitterBaseTask):
    """Output user-course grade matrix in libFM format."""
    ext = 'libfm'

    def output(self):
        fname = self.output_base_fname()
        guide = os.path.splitext(fname % 'guide')[0] + '.csv'
        return {
            'train': luigi.LocalTarget(fname % 'train'),
            'test': luigi.LocalTarget(fname % 'test'),
            'guide': luigi.LocalTarget(guide)
        }

    @property
    def train(self):
        try: return self._train
        except: self._train, self._test = self.split_data()
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self):
        try: return self._test
        except: self._train, self._test = self.split_data()
        return self._test

    @test.setter
    def test(self, test):
        self._test = test

    @property
    def term_range(self):
        """All terms for which prediction should be performed."""
        # Due to the backfilling, we must rely on the train filters to get the
        # last term in the training data.
        start = max([f.cohort_end for f in self.filters])
        end = int(self.test[~self.test.cohort.isnull()].cohort.max())
        return range(start + 1, end + 1)

    def handle_cold_start(self, test):
        """If requested, remove cold start, else do nothing."""
        if self.remove_cold_start:
            for key in ['sid', 'cid']:
                diff = np.setdiff1d(test[key].values, self.train[key].values)
                diff_mask = test[key].isin(diff)
                return test[~diff_mask]
        else:
            return test

    @property
    def write_libfm_data(self):
        def write_libfm_data(f, data):
            write_libfm(f, data, target='grdpts', cvals=self.cvals,
                        rvals=self.rvals)
        return write_libfm_data

    def run(self):
        """Produce train/test data for all-term prediction task."""
        # remove cold start records if requested
        test = self.test.copy()
        test = self.handle_cold_start(test)

        for name, dataset in zip(['train', 'test'], [self.train, test]):
            with self.output()[name].open('w') as f:
                self.write_libfm_data(f, dataset)

        # Write the term-to-id guide
        test = test.sort(('termnum'))
        test['rownum'] = np.arange(len(test))
        guide = test.groupby('termnum').max()['rownum']
        with self.output()['guide'].open('w') as f:
            guide.to_csv(f, index_label='termnum', header=True)


class LibFMNextTermInput(LibFMAllTermInput):
    """Output user-course grade matrix in libFM format."""

    def output(self):
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
        """Produce multiple train/test splits; one for each term to predict."""
        outputs = self.output()
        for termnum in self.term_range:  # includes (end term + 1)
            test = self.test[self.test.termnum == termnum]

            # remove cold start recordsif requested
            test = self.handle_cold_start(test)

            for name, dataset in zip(['train', 'test'], [self.train, test]):
                with outputs[termnum][name].open('w') as f:
                    self.write_libfm_data(f, dataset)
            self.transfer_term(termnum)  # modify train/test sets in place
            # intentionally skip writing the last time this is run

        # TODO: this code converts the same records to libFM format multiple
        # times. Each subsequent train set contains records in the last train
        # set. These could be cached to avoid all of the string format
        # conversion overhead.


if __name__ == "__main__":
    luigi.run()
