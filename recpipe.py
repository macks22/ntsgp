import os
import csv
import subprocess as sub

import luigi
import numpy as np
import pandas as pd

import test_params


class LuigiDataFile(luigi.Task):
    """Class to access files that already exist (no processing needed)."""
    data_fname = 'placeholder'

    def output(self):
        return luigi.LocalTarget(os.path.join('data', self.data_fname))


class StudentData(LuigiDataFile):
    data_fname = 'nsf_student.csv'

class AdmissionsData(LuigiDataFile):
    data_fname = 'nsf_admissions.csv'

class DegreesData(LuigiDataFile):
    data_fname = 'nsf_degrees.csv'

class CoursesData(LuigiDataFile):
    data_fname = 'nsf_courses.csv'


def fname_from_cname(cname):
    words = []
    chars = [cname[0]]
    for c in cname[1:]:
        if c.isupper():
            words.append(''.join(chars))
            chars = [c]
        else:
            chars.append(c)

    words.append(''.join(chars))
    return '-'.join(map(lambda s: s.lower(), words))


class BasicLuigiTask(luigi.Task):
    """Uses class name as output file."""
    ext = 'csv'

    def output(self):
        fname = fname_from_cname(self.__class__.__name__)
        return luigi.LocalTarget(os.path.join(
            'data', '%s.%s' % (fname, self.ext)))


class StudentIdMap(BasicLuigiTask):
    """Produce a contiguous id mapping for all students."""

    ids = ['id']

    def requires(self):
        return CoursesData()

    def run(self):
        with self.input().open() as f:
            courses = pd.read_csv(f)

        students = courses[self.ids].drop_duplicates().reset_index()[self.ids]
        with self.output().open('w') as f:
            students.to_csv(f, header='id', index_label='index', float_format='%.0f')


class CourseIdMap(BasicLuigiTask):
    """Produce a contiguous id mapping for all courses (DISC, CNUM, HRS)."""

    ids = ['DISC', 'CNUM', 'HRS']

    def requires(self):
        return CoursesData()

    def run(self):
        with self.input().open() as f:
            courses = pd.read_csv(f)

        # Get all unique courses, as specified by (DISC, CNUM, HRS).
        # we assume here that some courses have labs/recitations which
        # have the same (DISC, CNUM) but different number of HRS.
        triplets = courses[self.ids].drop_duplicates().reset_index()[self.ids]
        with self.output().open('w') as out:
            triplets.to_csv(out, index_label='index')


class InstructorIdMap(BasicLuigiTask):
    """Produce a contiguous id mapping for all instructors (LANME, FNAME)."""

    ids = ['INSTR_LNAME', 'INSTR_FNAME']

    def requires(self):
        return CoursesData()

    def run(self):
        with self.input().open() as f:
            courses = pd.read_csv(f)

        instr = courses[self.ids].drop_duplicates().reset_index()[self.ids]
        with self.output().open('w') as out:
            instr.to_csv(out, index_label='index', float_format='%.0f')


class OrdinalTermMap(BasicLuigiTask):
    """Produce an ordinal mapping (0...) for enrollment terms."""

    ids = ['TERMBNR']

    def requires(self):
        return CoursesData()

    def run(self):
        with self.input().open() as f:
            courses = pd.read_csv(f)

        terms = courses[self.ids].drop_duplicates().reset_index()[self.ids]
        with self.output().open('w') as out:
            terms.to_csv(out, index_label='index', float_format='%.0f')


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


class PreprocessedCourseData(BasicLuigiTask):
    """Clean up courses data to prepare for learning tasks."""

    def requires(self):
        return {'courses': CoursesData(),
                'admissions': AdmissionsData(),
                'StudentIdMap': StudentIdMap(),
                'CourseIdMap': CourseIdMap(),
                'InstructorIdMap': InstructorIdMap(),
                'OrdinalTermMap': OrdinalTermMap()}

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
        with self.input()['courses'].open() as f:
            courses = pd.read_csv(f)

        # fill in missing values for quality points
        courses.grdpts = courses.apply(self.fill_grdpts, axis=1)

        def map_ids(input_name, idname):
            klass = globals()[input_name]
            with self.input()[input_name].open() as f:
                idmap = pd.read_csv(f, index_col=0)

            idmap[idname] = idmap.index
            cols = list(courses.columns.values) + [idname]
            for col_name in klass.ids:
                cols.remove(col_name)

            for col_name in klass.ids:
                dtype = courses[col_name].dtype.type
                idmap[col_name] = idmap[col_name].values.astype(dtype)
            return courses.merge(idmap, how='left', on=klass.ids)[cols]

        # add student cohorts to data frame
        with self.input()['admissions'].open() as f:
            admiss = pd.read_csv(f, usecols=(0,1))

        with self.input()['OrdinalTermMap'].open() as f:
            idmap = pd.read_csv(f, index_col=0)

        admiss.columns = ['id', 'TERMBNR']
        idmap['cohort'] = idmap.index
        admiss = admiss.merge(idmap, how='left', on='TERMBNR')
        del admiss['TERMBNR']
        courses = courses.merge(admiss, how='left', on='id')

        # Replace course, student, instructor and term identifiers with
        # contiguous id mappings
        idmap = {'CourseIdMap': 'cid',
                 'StudentIdMap': 'sid',
                 'InstructorIdMap': 'iid',
                 'OrdinalTermMap': 'termnum'}

        for map_klass, idname in idmap.items():
            courses = map_ids(map_klass, idname)

        # remove unneeded columns
        unneeded = ['CRN', 'SECTNO', 'TITLE',
                    'class', 'instr_rank', 'instr_tenure']
        for col_name in unneeded:
            del courses[col_name]

        # Write cleaned up courses data.
        with self.output().open('w') as out:
            courses.to_csv(out, index=False)


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
    discard_nongrade = luigi.Parameter(
        default=True,
        description='drop W/S/NC grades from training data if True')
    backfill_cold_students = luigi.IntParameter(
        default=0,
        description="number of courses to backfill for cold-start students")
    backfill_cold_courses = luigi.IntParameter(
        default=0,
        description="number of courses to backfill for cold-start courses")
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

        # indicate whether cold-start backfilling was done for students/courses
        if self.remove_cold_start:
            parts.append('nocs')
        else:
            parts.append('scs%d' % self.backfill_cold_students)
            parts.append('ccs%d' % self.backfill_cold_courses)

        # include optional class-specific suffix
        if self.suffix:
            parts.append(self.suffix)

        fbase = os.path.join(self.base, '-'.join(parts))
        return '{}.%s.{}'.format(fbase, self.ext)


class DataSplitterBaseTask(UsesTrainTestSplit):
    """Functionality to split train/test data, no run method."""

    def requires(self):
        return PreprocessedCourseData()

    def read_data(self):
        with self.input().open() as f:
            data = pd.read_csv(f)

        # only keep most recent grade
        data = data.drop_duplicates(('sid','cid'), take_last=True)

        # remove records for missing grades
        data = data[~data['grdpts'].isnull()]
        return data

    def backfill(self, train, test, key, firstn):
        """Used to prevent cold-start records.

        :param DataFrame train: The training data.
        :param DataFrame test: The test data.
        :param str key: The key to backfill records for.
        :param int firstn: How many records to backfill for cold-starts.

        """
        if not firstn:  # specified 0 records for backfill
            return (train, test)

        diff = np.setdiff1d(test[key].values, train[key].values)
        diff_mask = test[key].isin(diff)
        diff_records = test[diff_mask]

        # figure out which records to transfer from test set to train set
        # some keys will have less records than specified
        gb = diff_records.groupby(key)
        counts = gb[key].transform('count')
        tokeep = counts - firstn
        tokeep[tokeep < 0] = 0

        # update train/test sets
        removing = gb.head(firstn)
        keeping = gb.tail(tokeep)
        test = pd.concat((test[~diff_mask], keeping))
        train = pd.concat((train, removing))
        return (train, test)

    def split_data(self):
        data = self.read_data()

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

        # remove W/S/NC from test set; it never makes sense to test on these
        toremove = ['W', 'S', 'NC']
        test = test[~test.GRADE.isin(toremove)]

        # optionally discard W/S/NC from train set
        if self.discard_nongrade:
            train = train[~train.GRADE.isin(toremove)]

        # if instructed to avoid student/course cold-start,
        # ensure all students/courses in the test set are also in the train set
        if not self.remove_cold_start:
            train, test = self.backfill(
                train, test, 'sid', self.backfill_cold_students)
            train, test = self.backfill(
                train, test, 'cid', self.backfill_cold_courses)

        return (train, test)


def write_triples(f, data, userid='sid', itemid='cid', rating='grdpts'):
    """Write a data file of triples (sparse matrix).

    :param str userid: Name of user id column (matrix rows).
    :param str itemid: Name of item id column (matrix cols).
    :param str rating: Name of rating column (matrix entries).
    """
    cols = [userid, itemid, rating]
    data.to_csv(f, sep='\t', header=False, index=False, columns=cols)


class UserCourseGradeTriples(DataSplitterBaseTask):
    """Produce a User x Course matrix with quality points as entries."""

    def run(self):
        """Write the train/test data in triple format."""
        for name, dataset in zip(['train', 'test'], self.split_data()):
            with self.output()[name].open('w') as f:
                write_triples(f, dataset)


def write_libfm(f, data, userid='sid', itemid='cid', rating='grdpts',
                timecol='', time_feat_num=0):
    """Write a data file of triples (sparse matrix). This assumes the column ids
    have already been offset by the max row id.

    :param str userid: Name of user id column (matrix rows).
    :param str itemid: Name of item id column (matrix cols).
    :param str rating: Name of rating column (matrix entries).
    :param int time_feat_num: Feature number for time attribute.
    :param str timecol: Name of temporal column.
    """
    # TimeSVD
    if time_feat_num:  # write time as categorical attribute
        def extract_row(series):
            return '%f %d:1 %d:1 %d:%d' % (
                series[rating], series[userid], series[itemid],
                time_feat_num, series[timecol])
    # time-aware BPTF model
    elif timecol:
        def extract_row(series):
            return '%f %d:1 %d:1 %d:1' % (
                series[rating], series[userid], series[itemid], series[timecol])
    # regularized SVD
    else:
        def extract_row(series):
            return '%f %d:1 %d:1' % (
                series[rating], series[userid], series[itemid])

    lines = data.apply(extract_row, axis=1)
    f.write('\n'.join(lines))


class UserCourseGradeLibFM(DataSplitterBaseTask):
    """Output user-course grade matrix in libFM format."""
    time = luigi.Parameter(
        default='',
        description='if empty; no time attributes, ' +
                    'cat = categorical encoding (TimeSVD), ' +
                    'bin = binary, one-hot encoding (BPTF)')
    task = luigi.Parameter(
        default='next',
        description='prediction task; next = next-term, all = all-terms')
    ext = 'libfm'

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
    def train(self):
        try: return self._train
        except: self._train, self._test = self.prep_data()
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self):
        try: return self._test
        except: self._train, self._test = self.prep_data()
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

    @property
    def suffix(self):
        return 'T%s' % self.time if self.time else ''

    def prep_data(self):
        """libFM has no notion of columns. It simply takes feature vectors with
        labels. So we need to re-encode the columns by adding the max row index.
        """
        train, test = self.split_data()
        max_row_idx = max(np.concatenate((test.sid.values, train.sid.values)))
        train.cid += max_row_idx
        test.cid += max_row_idx
        return (train, test)

    @property
    def write_libfm_data(self):
        """If time is included, as a feature, we need to specify how it will be
        written in the libFM format. The method being emulated changes based on
        how we choose to encode it. We multiplex based on time in the sense that
        we define a data writing function that writes the time data differently
        depending on what the user has specified. This function is returned.
        """
        # one-hot encoding; BTPF
        if self.time == 'bin':
            max_col_idx = max(
                np.concatenate((self.test.cid.values, self.train.cid.values)))

            def write_libfm_data(f, data):
                data.termnum += max_col_idx
                write_libfm(f, data, timecol='termnum')

        # categorical encoding; TimeSVD
        elif self.time == 'cat':
            max_col_idx = max(
                np.concatenate((self.test.cid.values, self.train.cid.values)))

            def write_libfm_data(f, data):
                write_libfm(f, data, timecol='termnum',
                            time_feat_num=max_col_idx + 1)

        # do not include time variables in output
        else:
            write_libfm_data = write_libfm

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
                return test[~diff_mask]
        else:
            return test

    def produce_all_term_data(self):
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

    def produce_next_term_data(self):
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

    def run(self):
        """Write the train/test data in libFM format."""
        if self.task == 'all':
            self.produce_all_term_data()
        else:
            self.produce_next_term_data()


if __name__ == "__main__":
    luigi.run()
