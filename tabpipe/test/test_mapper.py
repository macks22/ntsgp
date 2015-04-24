import os
import sys
import unittest

import pandas as pd
import numpy as np

from util import DataSetupTestCase

sys.path.append('../../')
from tabpipe.util import *
from tabpipe.core import DataTable
from tabpipe.mapper import (
    ColumnIdMapper, ValueSubber, Mapper
)


class TestColumnIdMapper(DataSetupTestCase):
    """Test the ColumnIdMapper Task."""
    pass


class TestValueSubber(DataSetupTestCase):
    """Test the ValueSubber Task."""
    pass


class TestMapper(DataSetupTestCase):
    """Test the Mapper Task."""

    dfile = 'cleaner-test-file.csv'
    dfiles = [dfile]

    data_sources = [{
        'grade': np.array(['A', 'B', 'C', 'A', 'B', 'C','F']),
        'status': np.array(['UG', 'UG', 'UG', 'UG', 'GRAD', 'GRAD', 'GRAD']),
        'id': np.random.choice(np.arange(10, 100), replace=False, size=7)
    }]

    def test_mapper_basic_case(self):
        """Test mapper for 2 string columns and one int column."""
        self.task = Mapper(
            table=self.dtables()[0],
            colnames=('grade', 'status', 'id'))
        self.task.run()
        df = self.task.read_output_table()

        # First column mapping; single character mapping.
        grades = np.array([0, 1, 2, 0, 1, 2, 3])
        check1 = (grades == df['grade'])
        self.assertTrue(check1.all())

        # Second column mapping; word mapping.
        status = np.array([0, 0, 0, 0, 1, 1, 1])
        check2 = (status == df['status'])
        self.assertTrue(check2.all())

        # Third column; numeric -> numeric.
        ids = np.array([0, 1, 2, 3, 4, 5, 6])
        check3 = (ids == df['id'])
        self.assertTrue(check3.all())

        self.task.delete_intermediates()


if __name__ == "__main__":
    unittest.main()
