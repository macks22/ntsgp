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
    pass


if __name__ == "__main__":
    unittest.main()
