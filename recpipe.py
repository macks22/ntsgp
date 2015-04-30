import os
import csv
import subprocess as sub

import luigi
import numpy as np
import pandas as pd

import test_params
from util import *
from writer import *


MAX_NUM_COHORTS = 14


class LuigiDataFile(luigi.ExternalTask):
    """Class to access files that already exist (no processing needed)."""
    data_fname = 'placeholder'
    def output(self):
        return luigi.LocalTarget(os.path.join('data', self.data_fname))


class BasicLuigiTask(luigi.Task):
    """Uses class name as output file."""
    ext = 'csv'
    def output(self):
        fname = fname_from_cname(self.__class__.__name__)
        return luigi.LocalTarget(os.path.join(
            'data', '%s.%s' % (fname, self.ext)))


class BuildIdMap(BasicLuigiTask):
    """Produce a contiguous id mapping for all ids."""
    ids = []

    def run(self):
        with self.input().open() as f:
            data = pd.read_csv(f)

        # prevent ids from being written as floats when we have nan values.
        isfloat = \
            [data[colname].apply(lambda val: isinstance(val, np.float)).any()
             for colname in self.ids]
        have_floats = np.array(isfloat).any()
        float_fmt = '%.6f' if have_floats else '%.0f'

        # Map sorted values to contiguous numerical indices.
        idmap = data[self.ids].sort(self.ids)\
                              .drop_duplicates()\
                              .reset_index()[self.ids]
        with self.output().open('w') as out:
            idmap.to_csv(out, index_label='index', float_format=float_fmt)


# All external data files.
DATA_SOURCES = {
    'courses': 'nsf_courses.csv',
    'students': 'nsf_student.csv',
    'demographics': 'nsf_demographics.csv',
    'admissions': 'nsf_admissions.csv',
    'degrees': 'nsf_degrees.csv'
}

# Build data source tasks; dynamically create class for each data source.
# After creation, place in new map, from source name to production task.
DATA_TASKS = {}
for src_name, fname in DATA_SOURCES.items():
    class_name = '%s%sData' % (src_name[0].upper(), src_name[1:])
    globals()[class_name] = \
        type(class_name, (LuigiDataFile,), {'data_fname': fname})
    DATA_TASKS[src_name] = globals()[class_name]


# WARNING: Does not currently support mapping to same colname!
IDMAPS = {
    'courses': {
        'sid': ['id'],
        'cid': ['DISC', 'CNUM', 'HRS'],
        'iid': ['INSTR_LNAME', 'INSTR_FNAME'],
        'termnum': ['TERMBNR'],
        'iclass': ['class'],
        'irank': ['instr_rank'],
        'itenure': ['instr_tenure'],
        'cdisc': ['DISC']
    },
    'demographics': {
        'srace': ['race'],
        'sex': ['SEX']
    },
    'students': {
        'major': ['PMAJR']
    },
    'admissions': {
        'zip': ['Permanent_Address_ZIP'],
        'hs': ['HS_CEEB_Code']
    },
    'degrees': {}
}

# Build id-mapping tasks at global scope.
IDMAP_TASKS = {}
for src_name, mapdict in IDMAPS.items():
    data_task = DATA_TASKS[src_name]
    base_name = '%s%s' % (src_name[0].upper(), src_name[1:])
    for attr, cols in mapdict.items():
        instance = data_task()
        # print src_name, attr, ','.join(cols), data_task, instance.output().path
        class_name = '%s%s%sMap' % (base_name, attr[0].upper(), attr[1:])
        globals()[class_name] = type(
            class_name, (BuildIdMap,),
            {'ids': cols,
             'reqtask': data_task,
             'requires': lambda self: self.reqtask()})
        IDMAP_TASKS[attr] = globals()[class_name]


# Alphabetical grade to quality points
# Guides consulted:
# https://www.gmu.edu/academics/catalog/0203/apolicies/examsgrades.html
# http://catalog.gmu.edu/content.php?catoid=15&navoid=1168
# https://registrar.gmu.edu/topics/special-grades/
grade2pts = {
    'A+':   4.0,
    'A':    4.0,
    'A-':   3.67,
    'B+':   3.33,
    'B':    3.00,
    'B-':   2.67,
    'C+':   2.33,
    'C':    2.00,
    'C-':   1.67,
    'D':    1.00,
    'F':    0.00,
    'IN':   0.00,    # Incomplete
    'S':    3.00,    # Satisfactory (passing; C and up, no effect on GPA)
    'NC':   1.00,    # No Credit (often C- and below)
    'W':    1.00,    # Withdrawal (does not affect grade)
    'NR':   np.nan,  # Not Reported (possibly honor code violation)
    'AU':   np.nan,  # Audit
    'REG':  np.nan,  # ?
    'IX':   np.nan,  # Incomplete Extension
    'IP':   np.nan,  # In Progress
    'nan':  np.nan,  # Unknown
    np.nan: np.nan   # Unknown (for iteration over Series)
}


def use_idmap(df, idmap, newname, oldcols, remove=False):
    """Use the given idmap to create a column with `newname` in the `df` by
    merging on `oldcols`.
    """
    tmpname = '_new_%s' % newname
    idmap[tmpname] = idmap.index
    df = df.merge(idmap, how='left', left_on=newname, right_on=oldcols)
    df[newname] = df[tmpname]
    del df[tmpname]
    if remove:
        for colname in oldcols:
            del df[colname]
    return df


def map_ids(df, idname, remove=False):
    """Map the set-categorical values in the df to numerical indices. Adds a new
    column to the DataFrame and returns the new frame. Optionally removes the
    columns used to produce the mapping. `idname` is the name of the new column
    which will contain the mapped values. This should be present in IDMAPS.
    """
    # Lookup id-mapping class in global table.
    klass = IDMAP_TASKS[idname]
    task = klass()

    # Shouldn't really need this here.
    if not task.complete():
        task.run()

    # Load idmap from file.
    with task.output().open() as f:
        idmap = pd.read_csv(f, index_col=0)

    # Create tmp column to store new column.
    tmpname = '_new_%s' % idname
    idmap[tmpname] = idmap.index

    # Ensure data types match up after merge.
    for colname in klass.ids:
        dtype = df[colname].dtype.type
        idmap[colname] = idmap[colname].values.astype(dtype)

    # Move values from tmp column to `idname` column and delete tmp.
    df = df.merge(idmap, how='left', on=klass.ids)
    df[idname] = df[tmpname]
    del df[tmpname]

    # Remove columns used to create mapping, if requeseted.
    if remove:
        for colname in klass.ids:
            del df[colname]

    return df


def extract_clevel(cnum):
    """Extract the course level from the course number."""
    if cnum == np.nan:
        return np.nan
    cnum = str(cnum).strip()
    if not cnum:
        return np.nan
    digits = filter(lambda c: c.isdigit(), cnum)
    if not digits:
        return np.nan
    return digits[0]


class PreprocessedData(BasicLuigiTask):
    """Clean up courses data to prepare for learning tasks."""

    attributes = {
        'sid': 1, 'cdisc': 0, 'cid': 0, 'iid': 1, 'termnum': 1,
        'iclass': 1, 'irank': 1, 'itenure': 1, 'zip': 1, 'hs': 1,
        'major': 1, 'srace': 1, 'sex': 1
    }
    cvals = attributes.keys()

    # Combine rvals from data and those produced from feature engineering.
    rvals = ['grdpts', 'age', 'hsgpa', 'sat', 'chrs', 'clevel'] + ['lterm_gpa',
            'lterm_cum_gpa', 'total_chrs', 'num_enrolled', 'lterm_cgpa',
            'lterm_cum_cgpa']

    # Finally, create dict of all data source tasks and attribute mapping tasks
    # to be required by this task.
    data_tasks = {src_name: task() for src_name, task in DATA_TASKS.items()}
    idmap_classes = [IDMAP_TASKS[attr] for attr in cvals]
    idmap_tasks = {klass.__name__: klass() for klass in idmap_classes}

    def requires(self):
        sources = self.data_tasks.copy()
        sources.update(self.idmap_tasks)
        return sources

    @property
    def grade2pts(self):
        return grade2pts

    def fill_grdpts(self, series):
        """Fill in missing values for grade quality points."""
        # TODO: we can fill in missing lab grades with lecture grades if we can
        # match them up.
        if series['GRADE'] != np.nan:
            return self.grade2pts[series['GRADE']]
        else:
            return series['grdpts']

    def run(self):
        courses_cols = ['id', 'TERMBNR', 'DISC', 'CNUM', 'GRADE', 'HRS',
                        'grdpts', 'INSTR_LNAME', 'INSTR_FNAME', 'class',
                        'instr_rank', 'instr_tenure']
        with self.input()['courses'].open() as f:
            courses = pd.read_csv(f, usecols=courses_cols)

        # fill in missing values for quality points
        courses.grdpts = courses.apply(self.fill_grdpts, axis=1)

        # Get course level from CNUM.
        courses['clevel'] = courses['CNUM'].apply(extract_clevel)

        # add student data first.
        students_cols = ['id', 'cohort', 'TERMBNR', 'PMAJR', 'term_earn_hrs']
        with self.input()['students'].open() as f:
            students = pd.read_csv(f, usecols=students_cols)

        data = courses.merge(students, how='left', on=('id', 'TERMBNR'))

        # add demographics data next
        with self.input()['demographics'].open() as f:
            demog = pd.read_csv(f)

        data = data.merge(demog, how='left', on='id')

        # add admissions data to data frame
        admiss_cols = ['id', 'cohort', 'Permanent_Address_ZIP', 'HSGPA',
                       'SAT_Total_1600', 'HS_CEEB_Code']
        with self.input()['admissions'].open() as f:
            admiss = pd.read_csv(f, usecols=admiss_cols)

        # Merge with admissions data on (id, cohort).
        data = data.merge(admiss, how='left', on=('id', 'cohort'))

        # Map set-categorical ids to contiguous numerical indices.
        for idname, remove_flag in self.attributes.items():
            data = map_ids(data, idname, remove=remove_flag)

        # Map cohort column values to same numerical index used for TERMBNR.
        with self.input()['CoursesTermnumMap'].open() as f:
            idmap = pd.read_csv(f, index_col=0)

        data = use_idmap(
            data, idmap, 'cohort', oldcols=['TERMBNR'], remove=True)

        # remove unneeded columns not deleted during mapping procedure
        unneeded = ['DISC', 'CNUM']
        for colname in unneeded:
            del data[colname]

        data.rename(columns={
            'HSGPA': 'hsgpa',
            'SAT_Total_1600': 'sat',
            'HRS': 'chrs',
            'ENTRY_AGE': 'age'
        }, inplace=True)

        # remove records for missing grades
        data = data[~data['grdpts'].isnull()]

        # only keep most recent grade
        data = data.sort(['termnum', 'sid'])
        data = data.drop_duplicates(('sid','cid'), take_last=True)

        # Feature engineering.
        data = self.engineer_features(data)

        # Write cleaned up data.
        with self.output().open('w') as out:
            data.to_csv(out, index=False)

        return data

    def engineer_features(self, data):
        """Engineer new features from the existing data."""

        # Compute quality points for each record.
        data['qpts'] = data['chrs'] * data['grdpts']

        # Compute total quality points per term.
        data['term_qpts'] = data.groupby(['sid', 'termnum'])\
                                 [['qpts']].transform('sum')
        tmp = data[['sid', 'termnum', 'term_qpts']]\
                .drop_duplicates(['sid', 'termnum'])\
                .sort(['sid', 'termnum'])
        tmp['total_qpts'] = tmp.groupby('sid')[['term_qpts']]\
                               .transform('cumsum')
        del tmp['term_qpts']
        data = data.merge(tmp, how='left', on=['sid', 'termnum'])

        # Next compute total hours earned each term and across terms.
        data['term_chrs'] = data.groupby(['sid', 'termnum'])\
                                 [['chrs']].transform('sum')
        tmp = data[['sid', 'termnum', 'term_chrs']]\
                .drop_duplicates(['sid', 'termnum'])\
                .sort(['sid', 'termnum'])
        tmp['total_chrs'] = tmp.groupby('sid')[['term_chrs']]\
                               .transform('cumsum')
        del tmp['term_chrs']
        data = data.merge(tmp, how='left', on=['sid', 'termnum'])

        # Now we can compute term gpa...
        data['term_gpa'] = data['term_qpts'] / data['term_chrs']

        # and the running gpa for each student.
        data['cum_gpa'] = data['total_qpts'] / data['total_chrs']

        # Finally, shift several attributes forward so the feature vectors
        # include information from the last term to use for predicting values in
        # the current term. Leave out quality points because gpa is a summary.
        merge_on = ['sid', 'termnum']
        tmp = data.drop_duplicates(merge_on).sort(merge_on)
        cols = ['term_gpa', 'term_chrs', 'cum_gpa', 'total_chrs']
        shifted = tmp.groupby('sid')[cols].shift(1)
        keep = ['lterm_gpa', 'lterm_chrs', 'lterm_cum_gpa', 'lterm_total_chrs']
        shifted.columns = keep
        keep += merge_on
        tmp = tmp.merge(shifted, how='left', right_index=True, left_index=True)
        tmp = tmp[keep]
        data = data.merge(tmp, how='left', on=merge_on)

        # Now we're done with student GPA features. Let's move on to course GPA,
        # AKA course difficulty as evidenced by student grdpts over time.

        # First, we add total # students enrolled at each term and across them.
        data['num_enrolled'] = data.groupby(['cid', 'termnum'])['cid']\
                                   .transform('count')

        # Add total number of students enrolled so far at each term.
        tmp = data[['cid', 'termnum', 'num_enrolled']]\
                .drop_duplicates(['cid', 'termnum'])\
                .sort(['cid', 'termnum'])
        tmp['total_enrolled'] = tmp.groupby('cid')[['num_enrolled']]\
                                   .transform('cumsum')
        del tmp['num_enrolled']
        data = data.merge(tmp, how='left', on=['cid', 'termnum'])

        # Now sum grdpts together for each term.
        data['term_grdpts_sum'] = data.groupby(['cid', 'termnum'])\
                                   [['grdpts']].transform('sum')
        tmp = data[['cid', 'termnum', 'term_grdpts_sum']]\
                .drop_duplicates(['cid', 'termnum'])\
                .sort(['cid', 'termnum'])
        tmp['total_grdpts_sum'] = tmp.groupby('cid')[['term_grdpts_sum']]\
                                    .transform('cumsum')
        del tmp['term_grdpts_sum']
        data = data.merge(tmp, how='left', on=['cid', 'termnum'])

        # Now we can compute course avg. gpa at each term...
        data['term_cgpa'] = data['term_grdpts_sum'] / data['num_enrolled']

        # and the running avg course gpa.
        data['cum_cgpa'] = data['total_grdpts_sum'] / data['total_enrolled']

        # Finally, shift some feature values forward one to make the previous
        # term's values accessible for prediction in the current term.
        merge_on = ['cid', 'termnum']
        tmp = data.drop_duplicates(merge_on).sort(merge_on)
        cols = ['term_cgpa', 'cum_cgpa', 'num_enrolled', 'total_enrolled']
        shifted = tmp.groupby('cid')[cols].shift(1)
        keep = ['lterm_cgpa', 'lterm_cum_cgpa', 'lterm_num_enrolled',
                'lterm_total_enrolled']
        shifted.columns = keep
        keep += merge_on
        tmp = tmp.merge(shifted, how='left', right_index=True, left_index=True)
        tmp = tmp[keep]
        data = data.merge(tmp, how='left', on=merge_on)

        return data


class TrainTestFilter(object):
    """Wrapper class to filter data to train/test sets using cohort/term."""
    term_max = MAX_NUM_COHORTS  # some number greater than max term id

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
        default='0-1',
        description='Specify how to split the train set from the test set.')
    discard_nongrade = luigi.Parameter(
        default=True,
        description='drop W/S/NC grades from training data if True')
    remove_cold_start = luigi.IntParameter(
        default=1,
        description="remove all cold-start records from test set")

    base = 'data'  # directory to write files to
    ext = 'tsv'    # final file extension for output files
    prefix = 'ucg' # prefix for all output files
    suffix = ''    # class-specific suffix that goes before ext on output names

    @property
    def filters(self):
        return [TrainTestFilter(filt) for filt in self.train_filters.split()]

    def output_base_fname(self):
        parts = [self.prefix] if self.prefix else []

        # parameter suffix part
        param_suffix = '-'.join([str(filt) for filt in self.filters])
        if param_suffix:
            parts.append(param_suffix)

        # indicate if W/S/NC grades are being included in train set
        if not self.discard_nongrade:
            parts.append('ng')

        # indicate whether cold-start records were removed for students/courses
        if self.remove_cold_start:
            parts.append('nocs')

        # include optional class-specific suffix
        if self.suffix:
            parts.append(self.suffix)

        fbase = os.path.join(self.base, '-'.join(parts))
        return '{}.%s.{}'.format(fbase, self.ext)


class UsesFeatures(UsesTrainTestSplit):
    """Allow selection of features via luigi cmdline params."""
    prev_cgrades = luigi.BoolParameter(
        default=False,
        description='use past course grades as features')

    # Copy possible feature names from the source data task.
    data_source = PreprocessedData()
    cvals = data_source.cvals[:]
    cvals.remove('cid')  # non-optional
    cvals.remove('sid')  # non-optional
    rvals = data_source.rvals[:]
    rvals.remove('grdpts')  # non-optional

    # TODO: get rid of featname in class namespace
    possible_features = cvals + rvals
    for featname in possible_features:
        locals()[featname] = luigi.BoolParameter(default=False)

    @property
    def cvals_to_write(self):
        return [cval for cval in self.cvals if getattr(self, cval, '')]

    @property
    def rvals_to_write(self):
        return [rval for rval in self.rvals if getattr(self, rval, '')]

    @property
    def features(self):
        return self.cvals_to_write + self.rvals_to_write

    @property
    def suffix(self):
        parts = [abbrev_names(self.features)]
        if self.prev_cgrades:
            parts.append('Pcgr')
        return '-'.join(parts)


class DataSplitterBaseTask(UsesFeatures):
    """Functionality to split train/test data, no run method."""

    def requires(self):
        return self.data_source

    def read_data(self):
        with self.input().open() as f:
            return pd.read_csv(f)

    def split_data(self):
        data = self.read_data()

        # now do train/test split; drop duplicates in case filters overlap
        train = pd.concat([f.train(data) for f in self.filters]).drop_duplicates()
        test = pd.concat([f.test(data) for f in self.filters]).drop_duplicates()

        # sometimes cohorts have nan values, and other times students from later
        # cohorts take courses before they've officially enrolled.
        start = max([f.cohort_end for f in self.filters])
        oddball_mask = test.termnum <= start
        train = pd.concat((train, test[oddball_mask]))
        test = test[~oddball_mask]

        # remove W/S/NC from test set; it never makes sense to test on these
        toremove = ['W', 'S', 'NC']
        test = test[~test.GRADE.isin(toremove)]

        # optionally discard W/S/NC from train set
        if self.discard_nongrade:
            train = train[~train.GRADE.isin(toremove)]

        return (train, test)


class UserCourseGradeLibFM(DataSplitterBaseTask):
    """Output user-course grade matrix in libFM format."""
    task = luigi.Parameter(
        default='next',
        description='prediction task; next = next-term, all = all-terms')
    ext = 'libfm'

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

    def all_term_output(self):
        fname = self.output_base_fname()
        guide = os.path.splitext(fname % 'guide')[0] + '.csv'
        return {
            'train': luigi.LocalTarget(fname % 'train'),
            'test': luigi.LocalTarget(fname % 'test'),
            'guide': luigi.LocalTarget(guide)
        }

    def next_term_output(self):
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

    def output(self):
        if self.task == 'all':
            return self.all_term_output()
        else:
            return self.next_term_output()

    @property
    def term_range(self):
        """All terms for which prediction should be performed."""
        start = max([f.cohort_end for f in self.filters])
        end = MAX_NUM_COHORTS
        return range(start + 1, end + 1)

    @property
    def write_libfm_data(self):
        def write_libfm_data(ftrain, ftest, train, test):
            write_libfm(ftrain, ftest, train, test, target='grdpts',
                        userid='sid', itemid='cid', cvals=self.cvals_to_write,
                        rvals=self.rvals_to_write,
                        prev_cgrades=self.prev_cgrades)
        return write_libfm_data

    def transfer_term(self, termnum):
        """Move data for the given term from the test set to the train set."""
        tomove_mask = self.test.termnum == termnum
        self.train = pd.concat((self.train, self.test[tomove_mask]))
        self.test = self.test[~tomove_mask]

    def handle_cold_start(self, test):
        """If requested, remove cold start, else do nothing."""
        if self.remove_cold_start:
            for key in ['sid', 'cid']:
                diff = np.setdiff1d(test[key].values, self.train[key].values)
                diff_mask = test[key].isin(diff)
                test = test[~diff_mask]
        return test

    def produce_all_term_data(self):
        """Produce train/test data for all-term prediction task."""
        # remove cold start records if requested
        test = self.test.copy()
        test = self.handle_cold_start(test)

        outputs = self.output()
        trainf, testf = outputs['train'], outputs['test']
        with trainf.open('w') as ftrain, testf.open('w') as ftest:
            self.write_libfm_data(ftrain, ftest, self.train, test)

        # Write the term-to-id guide
        test = test.sort(('termnum'))
        test['rownum'] = np.arange(len(test))
        guide = test.groupby('termnum').max()['rownum']
        with self.output()['guide'].open('w') as f:
            guide.to_csv(f, index_label='termnum', header=True)

    def produce_next_term_data(self):
        """Produce multiple train/test splits; one for each term to predict."""
        outputs = self.output()
        for termnum in self.term_range:  # includes (end term + 1)
            test = self.test[self.test.termnum == termnum]

            # remove cold start recordsif requested
            test = self.handle_cold_start(test)

            term_outputs = outputs[termnum]
            trainf, testf = term_outputs['train'], term_outputs['test']
            with trainf.open('w') as ftrain, testf.open('w') as ftest:
                self.write_libfm_data(ftrain, ftest, self.train, test)

            self.transfer_term(termnum)  # modify train/test sets in place
            # intentionally skip writing the last time this is run

        # TODO: this code converts the same records to libFM format multiple
        # times. Each subsequent train set contains records in the last train
        # set. These could be cached to avoid all of the string format
        # conversion overhead.

    def run(self):
        """Write the train/test data in libFM format."""
        if self.task == 'all':
            self.produce_all_term_data()
        else:
            self.produce_next_term_data()


if __name__ == "__main__":
    luigi.run()
