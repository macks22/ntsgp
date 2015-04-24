import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.append('../')
from tabpipe import cleaners
from tabpipe import filters
from tabpipe.util import *
from tabpipe.core import DataTable
from tabpipe.cleaner import (
    ColumnScrubber, ColumnReplacer, RowFilterer, RowRemover, Cleaner
)


class DataSetupTestCase(unittest.TestCase):
    """Includes basic data set up and clean up methods."""

    dfiles = []  # list of string, to save data tables to
    data_sources = []  # list of dictionaries, to be converted to DataFrames
    task = None

    @classmethod
    def dtables(cls):
        return [DataTable(dfile) for dfile in cls.dfiles]

    @classmethod
    def iter_sources(cls):
        for item in zip(cls.dfiles, cls.data_sources):
            yield item

    @classmethod
    def setUpClass(cls):
        """Write DataFrame to be used for testing."""
        if len(cls.data_sources) != len(cls.dfiles):
            raise ValueError("More data files than data sources.")

        for fname, data in cls.iter_sources():
            df = pd.DataFrame(data)
            df.to_csv(fname, index=True)

    @classmethod
    def tearDownClass(cls):
        """Delete csv file written in set up."""
        for dfile in cls.dfiles:
            try:
                os.remove(dfile)
            except OSError:
                pass

    def tearDown(self):
        """Delete any files written by tasks during testing."""
        if self.task:
            try:
                os.remove(self.task.output().path)
            except OSError:
                pass


class TestColumnScrubber(DataSetupTestCase):
    """Test ColumnScrubber Task."""

    dfile = 'column-scrubber-test-file.csv'
    dfiles = [dfile]

    data_sources = [{
        'name': [' John ', 'Jack ', 'Jill ', 'Sally'],
        'age': [21, np.nan, 15, 47],
        'zip': [22030, 22030, 15354, np.nan]
    }]

    def test_fill_nan_using_mode(self):
        """Fill nan values with the mode of the column."""
        self.task = ColumnScrubber(
            table=self.dtables()[0], colnames='zip',
            func=cleaners.fill_nan_with_mode)
        self.task.run()
        df = self.task.read_output_table()
        self.assertEqual(df['zip'].ix[3], 22030)

    def test_fill_using_other_cols(self):
        """Fill nan values with values from other columns."""
        colname = 'age'
        self.task = ColumnScrubber(
            table=self.dtables()[0], colnames=(colname, 'zip'),
            func=cleaners.fill_nan_with_other_col)
        df_before = self.task.read_input_table()
        self.task.run()
        df = self.task.read_output_table()

        nan_mask = df_before[colname].isnull()
        check = df[colname][nan_mask] == df_before['zip'][nan_mask]
        self.assertTrue(check.sum().all())

    def test_strip_text_column(self):
        """Strip whitespace off text column."""
        colname ='name'
        self.task = ColumnScrubber(
            table=self.dtables()[0], colnames=colname,
            func=cleaners.strip_text_column)
        self.task.run()
        df = self.task.read_output_table()

        ws = [' ', '\t', '\n', '\r']
        mask = pd.Series([False] * len(df))
        for char in ws:
            mask = mask | (df[colname].str.startswith(char))
            mask = mask | (df[colname].str.endswith(char))

        self.assertEqual(mask.sum(), 0)


class TestColumnReplacer(DataSetupTestCase):
    """Test ColumnReplacer Task."""

    dfile = 'column-replacer-test-file.csv'
    replace_file = 'replacement-column.csv'
    dfiles = [dfile, replace_file]

    data_sources = [
        {'name': [' John ', 'Jack ', 'Jill ', 'Sally'],
         'age': [21, np.nan, 15, 47],
         'zip': [22030, 22030, 15354, np.nan]},
        {'name': ['001', '002', '003', '004']}
    ]

    def test_replace_column(self):
        """Replace a column."""
        colname = 'name'
        dtable, rtable = self.dtables()
        self.task = ColumnReplacer(
            table=dtable, replacement=rtable, colnames=colname)
        df_before = self.task.read_input_table()
        self.task.run()

        # Now all values in the name column should be equal to those in the name
        # column of the replacement table.
        df = self.task.read_output_table()
        with self.task.input()['replacement'].open() as f:
            rdf = pd.read_csv(f, index_col=0)

        check = (df[colname] == rdf[colname])
        self.assertTrue(check.all())


class TestRowCleaning(DataSetupTestCase):
    """Data and tasks for testing row filtering/removing."""

    dfile = 'row-cleaning-test-file.csv'
    dfiles = [dfile]

    data_sources = [{
        'missing': np.array(
            [np.nan, 1, np.nan, 2, np.nan, 3, np.nan, 4, np.nan]),
        'numeric': np.array(
            [-20, -15, -10, -5, 0, 5, 10, 15, 20])
    }]

    @classmethod
    def nan_filter_task(cls, colname='missing'):
        return RowFilterer(
            table=cls.dtables()[0], colnames=colname,
            func=filters.filter_out_nan_values, name='no-nan')

    @classmethod
    def numeric_range_filter_task(cls, colname='numeric', bounds=(-10, 10)):
        func = lambda df, colnames: filters.filter_to_numeric_range(
            df, colnames, bounds)
        return RowFilterer(
            table=cls.dtables()[0], colnames=colname,
            func=func, name='num-range')


class TestRowFilterer(TestRowCleaning):
    """Test RowFilterer Task."""

    def test_filter_out_rows_with_nan_values(self):
        """Filter out rows that have nan values in some column."""
        colname = 'missing'
        self.task = self.nan_filter_task(colname)
        self.task.run()

        # There should no longer be nan values in `colname`.
        df = self.task.read_output_table()
        indices = df.index.values

        # all nan values should be marked for removal.
        source = self.data_sources[0][colname]
        actual = np.isnan(source).nonzero()[0]
        check = indices == actual
        self.assertTrue(check.all())

    def test_filter_to_rows_in_numeric_range(self):
        """Filter out all rows outside some numeric range."""
        colname = 'numeric'
        lower, upper = (-10, 10)
        bounds = (lower, upper)
        self.task = self.numeric_range_filter_task(colname, bounds)
        self.task.run()

        # There should no longer be values < -10 or > 10 in `colname`.
        df = self.task.read_output_table()
        indices = df.index.values

        # All values beyond bounds should be marked for removal.
        source = self.data_sources[0][colname]
        actual = ((source < lower) | (source > upper)).nonzero()[0]
        check = indices == actual
        self.assertTrue(check.all())


class TestRowRemover(TestRowCleaning):
    """Test RowRemover Task."""

    def test_remove_rows_two_filters(self):
        """Remove rows using two RowFilterers."""
        bounds = (-10, 10)
        dtable = self.dtables()[0]
        filter_tables = [
            self.nan_filter_task(colname='missing'),
            self.numeric_range_filter_task(colname='numeric', bounds=bounds)
        ]
        self.task = RowRemover(
            table=dtable, filter_tables=filter_tables)
        schedule_task(self.task)

        # All rows with nan values in `missing` col and numeric values < -10 or
        # > 10 in `numeric` column should now be gone.
        df = pd.DataFrame(self.data_sources[0])
        mask = ((df['missing'].isnull()) |     # is missing
                (df['numeric'] < bounds[0]) |  # below lower bound
                (df['numeric'] > bounds[1]))   # above upper bound
        source = df[~mask]  # meets all conditions filters should handle.
        actual = source.index.values

        df = self.task.read_output_table()
        indices = df.index.values
        self.assertEqual(len(indices), len(actual))

        check = indices == actual
        self.assertTrue(check.all())

        # tear down
        for task in self.task.deps():
            try:
                os.remove(task.output().path)
            except OSError:
                pass


class TestCleaner(DataSetupTestCase):
    """Test Cleaner Task."""

    dfile = 'cleaner-test-file.csv'
    dfiles = [dfile]

    data_sources = [{
        # provide a useful joining column that is not the index
        'id': np.arange(10),
        # missing values that need to be filled
        'gpa': np.array([np.nan, 1, 2, np.nan, 4, 0, 1, np.nan, 3, np.nan]),
        # another column to fill missing values with
        'grade': np.array(['F', 'D', 'C', 'B', 'A', 'F', 'D', 'C', 'B', 'A']),
        # numeric bounds filtering
        'hsgpa': np.array([3.5, 2.0, 2.5, 2.75, 4.0, 1200, -13.3, 4.3, 1.5, 5.0])
    }]

    def test_cleaning_student_data(self):
        """Run cleaning process on student data."""
        bounds = (0, 5)
        self.task = Cleaner(
            table=self.dtables()[0], outname='cleaned-student-data.csv',
            scrubbers = [
                {'colnames': ['gpa', 'grade'],
                 'func': cleaners.fill_nan_with_2col_1to1_map}
            ],
            filterers = [
                {'name': 'bound-hsgpa',
                 'colnames': 'hsgpa',
                 'func': lambda df, colns: \
                     filters.filter_to_numeric_range(df, colns, bounds)}
            ],
            primary_id = 'id'
        )
        self.task.run()
        df = self.task.read_output_table()

        source = self.data_sources[0]
        mask = (source['hsgpa'] < bounds[0]) | (source['hsgpa'] > bounds[1])
        indices_removed = mask.nonzero()[0]

        check1 = df.index.isin(indices_removed)
        self.assertFalse(check1.any())

        gpacol = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # nan values filled
        indices_removed.sort()
        for idx in indices_removed:
            gpacol.pop(idx)
            indices_removed -= 1

        gpacol = np.array(gpacol)
        check2 = (df['gpa'] == gpacol)
        self.assertTrue(check2.all())

        self.task.delete_intermediates()


if __name__ == "__main__":
    unittest.main()
