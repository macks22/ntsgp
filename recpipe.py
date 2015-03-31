import os
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

    def run(self):
        with self.input()['courses'].open() as f:
            courses = pd.read_csv(f)

        # fill in missing values for quality points
        # TODO: we can fill in missing lab grades with lecture grades if we can
        # match them up.
        def fill_grdpts(series):
            if series['GRADE'] != np.nan:
                return grade2pts[series['GRADE']]
            else:
                return series['grdpts']

        courses.grdpts = courses.apply(fill_grdpts, axis=1)

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


class DataSplitterBaseTask(luigi.Task):
    """Functionality to split train/test data, no run method."""
    train_filters = luigi.Parameter(default='0-14')
    discard_nongrade = luigi.Parameter(default=True)

    ext = 'tsv'
    suffix = ''

    def __init__(self, *args, **kwargs):
        super(DataSplitterBaseTask, self).__init__(*args, **kwargs)
        self.filters = \
            [TrainTestFilter(filt) for filt in self.train_filters.split()]

    def requires(self):
        return PreprocessedCourseData()

    def output(self):
        param_suffix = '-'.join([str(filt) for filt in self.filters])
        ng_suffix = 'ng' if self.discard_nongrade else ''
        parts = [self.suffix, ng_suffix]
        suffix = '-'.join(parts)
        if suffix and not suffix.startswith('-'):
            suffix = '-' + suffix

        base = 'data/ucg-{}{}.%s.{}'.format(param_suffix, suffix, self.ext)
        train = base % 'train'
        test =  base % 'test'
        return {
            'train': luigi.LocalTarget(train),
            'test': luigi.LocalTarget(test)
        }

    def read_data(self):
        with self.input().open() as f:
            data = pd.read_csv(f)

        # only keep most recent grade
        data = data.drop_duplicates(('sid','cid'), take_last=True)

        # remove records for missing grades
        data = data[~data['grdpts'].isnull()]
        return data

    def split_data(self):
        data = self.read_data()

        # sort data by term number, then by student id
        data = data.sort(['termnum', 'sid'])

        # now do train/test split
        train = pd.concat([f.train(data) for f in self.filters]).drop_duplicates()
        test = pd.concat([f.test(data) for f in self.filters]).drop_duplicates()

        # remove W/S/NC from test set
        toremove = ['W', 'S', 'NC']
        test = test[~test.GRADE.isin(toremove)]

        # optionally discard W/S/NC from train set
        if self.discard_nongrade:
            train = train[~train.GRADE.isin(toremove)]

        # ensure all classes in the test set are also in the training set
        diff = np.setdiff1d(test['cid'].values, train['cid'].values)
        diff_mask = test['cid'].isin(diff)
        diff_courses = test[diff_mask]

        # figure out which records to transfer from test set to train set
        topn = 3
        gb = diff_courses.groupby('cid')
        counts = gb['cid'].transform('count')
        tokeep = counts - topn
        tokeep[tokeep < 0] = 0

        # update train/test sets
        removing = gb.head(topn)
        keeping = gb.tail(tokeep)
        test = pd.concat((test[~diff_mask], keeping))
        train = pd.concat((train, removing))
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
        train, test = self.split_data()

        # write the train/test data
        with self.output()['train'].open('w') as f:
            write_triples(f, train)

        with self.output()['test'].open('w') as f:
            write_triples(f, test)


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
    time = luigi.Parameter(default='')  # include time attributes
    ext = 'libfm'

    def __init__(self, *args, **kwargs):
        super(UserCourseGradeLibFM, self).__init__(*args, **kwargs)
        if self.time:
            self.suffix = 'time-%s' % self.time

    def run(self):
        train, test = self.split_data()

        # libFM has no notion of columns, it simply takes feature vectors with
        # labels. So we need to re-encode the columns by adding the max row
        # index.
        max_row_idx = max(np.concatenate((test.sid.values, train.sid.values)))
        train.cid += max_row_idx
        test.cid += max_row_idx

        # If time is included, calculate feature number for categorical feature
        if self.time == 'bin':
            max_col_idx = max(
                np.concatenate((test.cid.values, train.cid.values)))
            train.termnum += max_col_idx
            test.termnum += max_col_idx
            def write_libfm_data(f, data):
                write_libfm(f, data, timecol='termnum')
        elif self.time == 'cat':  # categorical, TimeSVD
            max_col_idx = max(
                np.concatenate((test.cid.values, train.cid.values)))
            def write_libfm_data(f, data):
                write_libfm(f, data, timecol='termnum',
                            time_feat_num=max_col_idx + 1)
        else:
            write_libfm_data = write_libfm

        # write the train/test data
        with self.output()['train'].open('w') as f:
            write_libfm_data(f, train)

        with self.output()['test'].open('w') as f:
            write_libfm_data(f, test)


class RunLibFM(luigi.Task):
    train_filters = luigi.Parameter(
        description='Specify how to split the train set from the test set.')
    time = luigi.Parameter(
        default='',
        description='cat=categorical, bin=binary-encoded, default=time unused')
    discard_nongrade = luigi.Parameter(
        default=True,
        description='drop W/S/NC grades from training data if True')
    iterations = luigi.IntParameter(
        default=200,
        description='number of iterations to use for learning')
    init_stdev = luigi.FloatParameter(
        default=0.3,
        description='initial std of Gaussian spread; higher can be faster')
    use_bias = luigi.BoolParameter(
        default=False,
        description='use global and per-feature bias terms if True')
    dim_start = luigi.IntParameter(
        default=5,
        description='start of dimension range to produce results for')
    dim_end = luigi.IntParameter(
        default=20,
        description='end of dimension range to produce results for, inclusive')

    def __init__(self, *args, **kwargs):
        super(RunLibFM, self).__init__(*args, **kwargs)
        self.filters = \
            [TrainTestFilter(filt) for filt in self.train_filters.split()]

    def requires(self):
        return UserCourseGradeLibFM(
            train_filters=self.train_filters, time=self.time,
            discard_nongrade=self.discard_nongrade)

    def output(self):
        base = 'outcomes'
        parts = []

        # train/test data filtering part
        param_suffix = '-'.join([str(filt) for filt in self.filters])
        if param_suffix:
            parts.append(param_suffix)

        # nongrade part (did we include W/S/NC grades?)
        if self.discard_nongrade:
            parts.append('ng')

        # time information part
        if self.time:
            parts.append('time')
            parts.append(self.time)

        # number of iterations part
        parts.append(str(self.iterations))

        # initial standard deviation part (init_stdev)
        parts.append(str(self.init_stdev))

        # bias terms part
        if self.use_bias:
            parts.append('b')

        parts.append('out.txt')
        fname = os.path.join(base, '-'.join(parts))
        return luigi.LocalTarget(fname)

    def run(self):
        train = self.input()['train'].path
        test = self.input()['test'].path

        results = test_params.test_dim(
            self.dim_start, self.dim_end,
            train, test, self.iterations,
            std=self.init_stdev, bias=self.use_bias)

        with self.output().open('w') as f:
            output = '\n'.join(['\t'.join(result) for result in results])
            f.write(output)

# TODO: currently does not work; some issue with the conversion script being
# called using Popen... not sure why, but it breaks when reading input files
class UserCourseGradeLibFMFaulty(luigi.Task):
    """Convert triples to libFM format."""
    train_filters = luigi.Parameter(default='0-14')

    script_path = 'libfm-1.42.src/scripts/triple_format_to_libfm.pl'

    def requires(self):
        return UserCourseGradeTriples(train_filters=self.train_filters)

    def output(self):
        fnames = [f.path for f in self.input().values()]
        parts = map(os.path.splitext, fnames)
        outnames = ['%s.libfm' % base for base, ext in parts]
        targets = map(luigi.LocalTarget, outnames)
        return targets

    def run(self):
        # first convert the files to the libFM format
        infiles = self.input()
        inputs = [infiles['train'].path, infiles['test'].path]
        args = [
            self.script_path,
            '-in', ','.join(inputs),
            '-target', '2',
            '-separator', '"\\t"'
        ]
        print ' '.join(args)
        proc = sub.Popen(args, stdout=sub.PIPE)
        print proc.communicate()[0]

        # now clean up the file extensions
        parts = map(os.path.splitext, inputs)
        outnames = ['%s%s.libfm' % (base, ext) for base, ext in parts]
        fixed = [f.path for f in self.output()]

        for old, new in zip(outnames, fixed):
            print 'moving %s to %s' % (old, new)
            os.rename(old, new)


if __name__ == "__main__":
    luigi.run()
