import sys
import unittest

import pandas as pd
import numpy as np

sys.path.append('../../')
from tabpipe.util import *
from tabpipe.core import DataTable, ColumnReplacer, TableMerger

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


class TestTableMerger(DataSetupTestCase):
    """Test TableMerger Task."""

    dfiles = ['table%d' % i for i in range(3)]

    data_sources = [
        {'id': [0, 1, 2, 3, 4],
         'sat': [1250, 1200, 1000, 1600, 800]},
        {'id': [0, 1, 2, 5, 6],
         'gpa': [3.0, 3.0, 2.0, 4.0, 1.5]},
        {'id': [0, 0, 1, 1, 2],
         'cid': [0, 1, 0, 1, 0]}
    ]

    def test_merge_3_tables_same_id(self):
        """Merge 3 tables on the same, non-index id."""
        dt1, dt2, dt3 = self.dtables()
        idname = 'id'
        self.task = TableMerger(
            tables=[
                {'table': dt1, 'id': idname},
                {'table': dt2, 'id': idname},
                {'table': dt3, 'id': idname}
            ],
            outname='student-data.csv'
        )
        self.task.run()
        df = self.task.read_output_table()

        # Make sure all columns ended up in final table.
        colnames = np.array(['sat', 'gpa', 'cid', 'id'])
        check1 = np.in1d(colnames, df.columns.values)
        self.assertTrue(check1.all())

        # Make sure all ids are represented.
        ids = reduce(lambda x,y: x + y,
                     [src['id'] for src in self.data_sources])
        ids = np.array(list(set(ids)))
        check2 = np.in1d(ids, df[idname].values)
        self.assertTrue(check2.all())


if __name__ == "__main__":
    unittest.main()
