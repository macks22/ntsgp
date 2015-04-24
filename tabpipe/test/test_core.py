import sys
import unittest

import pandas as pd
import numpy as np

sys.path.append('../../')
from tabpipe.util import *
from tabpipe.core import DataTable, ColumnReplacer

from util import DataSetupTestCase


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


if __name__ == "__main__":
    unittest.main()
