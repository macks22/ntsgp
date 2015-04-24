import os

import luigi
import pandas as pd

from util import *
from core import MultipleTableTransform


class TableMerger(MultipleTableTransform):
    """Merge multiple DataFrames together."""
    on = luigi.Parameter(
        default=None,
        description='name of column (or csv string of names) to merge on')

    def run(self):
        """Merge all input data frames on the given colname(s)."""
        on = parse_colnames(self.on)
