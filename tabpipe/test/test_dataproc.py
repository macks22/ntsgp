import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.append('../')
from tabpipe import cleaners
from tabpipe.dataproc import DataTable, ColumnScrubber, ColumnReplacer


class TestColumnScrubber(unittest.TestCase):
    """Test ColumnScrubber Task."""

    dfile = 'column-scrubber-test-file.csv'
    dtable = DataTable(dfile)
    task = None

    @classmethod
    def setUpClass(cls):
        """Write DataFrame to be used for testing."""
        data = {
            'name': [' John ', 'Jack ', 'Jill ', 'Sally'],
            'age': [21, np.nan, 15, 47],
            'zip': [22030, 22030, 15354, np.nan]
        }
        df = pd.DataFrame(data)
        df.to_csv(cls.dfile, index=True)

    @classmethod
    def tearDownClass(cls):
        """Delete csv file written in set up."""
        os.remove(cls.dfile)

    def tearDown(self):
        try:
            os.remove(self.task.output().path)
        except OSError:
            pass

    def test_fill_nan_using_mode(self):
        """Fill nan values with the mode of the column."""
        self.task = ColumnScrubber(
            table=self.dtable, colname='zip', func=cleaners.fill_nan_with_mode)
        self.task.run()
        df = self.task.read_output_table()
        self.assertEqual(df['zip'].ix[3], 22030)

    def test_fill_using_other_cols(self):
        """Fill nan values with values from other columns."""
        colname = 'age'
        self.task = ColumnScrubber(
            table=self.dtable, colname=colname, othercols='zip',
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
            table=self.dtable, colname=colname,
            func=cleaners.strip_text_column)
        self.task.run()
        df = self.task.read_output_table()

        ws = [' ', '\t', '\n', '\r']
        mask = pd.Series([True] * len(df))
        for char in ws:
            mask = mask & (df[colname].str.startswith(char))
            mask = mask & (df[colname].str.endswith(char))

        self.assertEqual(mask.sum(), 0)


class TestColumnReplacer(unittest.TestCase):
    """Test ColumnReplacer Task."""

    dfile = 'column-replacer-test-file.csv'
    replace_file = 'replacement-column.csv'
    dtable = DataTable(dfile)
    rtable = DataTable(replace_file)
    task = None

    @classmethod
    def setUpClass(cls):
        """Create an original table and a replacement column."""
        data = {
            'name': [' John ', 'Jack ', 'Jill ', 'Sally'],
            'age': [21, np.nan, 15, 47],
            'zip': [22030, 22030, 15354, np.nan]
        }
        replacement = {'name': ['001', '002', '003', '004']}

        for fname, src in zip(
            [cls.dfile, cls.replace_file], [data, replacement]):
            df = pd.DataFrame(data)
            df.to_csv(fname, index=True)

    @classmethod
    def tearDownClass(cls):
        """Delete csv file written in set up."""
        for fname in [cls.dfile, cls.replace_file]:
            os.remove(fname)

    def tearDown(self):
        try:
            os.remove(self.task.output().path)
        except OSError:
            pass

    def test_replace_column(self):
        """Replace a column."""
        colname = 'name'
        self.task = ColumnReplacer(
            table=self.dtable, replacement=self.rtable, colname=colname)
        df_before = self.task.read_input_table()
        self.task.run()

        # Now all values in the name column should be equal to those in the name
        # column of the replacement table.
        df = self.task.read_output_table()
        with self.task.input()['replacement'].open() as f:
            rdf = pd.read_csv(f, index_col=0)

        check = (df[colname] == rdf[colname])
        self.assertTrue(check.all())


class TestRowFilterer(unittest.TestCase):
    """Test RowFilterer Task."""
    pass


class TestRowRemover(unittest.TestCase):
    """Test RowRemover Task."""
    pass


class TestCleaner(unittest.TestCase):
    """Test Cleaner Task."""
    pass


if __name__ == "__main__":
    unittest.main()
