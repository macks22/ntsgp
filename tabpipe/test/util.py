import os
import sys
import unittest

import pandas as pd

sys.path.append('../../')
from tabpipe.util import *
from tabpipe.core import DataTable


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

