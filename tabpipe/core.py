import os
import hashlib

import luigi
import pandas as pd


class TableMixin(object):
    """Useful table properties."""

    @property
    def tname(self):
        path = self.table.output().path
        return os.path.basename(os.path.splitext(path)[0])

    @property
    def thash(self):
        return hashlib.sha1(self.tname).hexdigest()


class DataTable(luigi.ExternalTask, TableMixin):
    """Wraps up external source files for data tables."""
    fname = luigi.Parameter(description='filename for data table')

    @property
    def table(self):
        return self

    def output(self):
        return luigi.LocalTarget(self.fname)


class TableBase(luigi.Task, TableMixin):
    """Miscellaneous table manipulation methods."""
    savedir = luigi.Parameter(
        default='',
        description='directory to save data files to; default is cwd')
    colnames = luigi.Parameter(
        description='name of column to scrub is first, then any others')
    outname = luigi.Parameter(
        description='filename to output transformed table to')

    @property
    def multiple_columns(self):
        """True if `colnames` is an interable, else False."""
        return hasattr(self.colnames, '__iter__')

    @property
    def cols(self):
        """Colnames to use as function input; list of 1+ strings."""
        return list(self.colnames) if self.multiple_columns else [self.colnames]

    @property
    def usecols(self):
        """Columns to use for reading data tables. Always assume index."""
        return [0] + self.cols

    @property
    def colname(self):
        """Main colname to use for table output and function input."""
        return self.colnames[0] if self.multiple_columns else self.colnames

    def read_output_table(self):
        """Only index and `colname` are written to output."""
        with self.output().open() as f:
            return pd.read_csv(f, index_col=0)

    def read_input_table(self):
        """Read only the columns which will be used."""
        input = self.input()
        if hasattr(input, 'keys'):
            infile = input['table']
        elif hasattr(input, '__iter__'):
            infile = input[0]
        else:
            infile = input

        with infile.open() as f:
            return pd.read_csv(f, index_col=0, usecols=self.usecols)

    def output(self):
        """Output to`savedir/outname`."""
        return luigi.LocalTarget(os.path.join(self.savedir, self.outname))

    run = NotImplemented


class TableTransform(TableBase):
    """Takes a task that outputs a table, transforms the table, writes it."""
    table = luigi.Parameter(
        description='task that outputs table to transform')

    def requires(self):
        return self.table


class MultipleTableTransform(TableBase):
    """Takes several tables and outputs one table."""
    tables = luigi.Parameter(
        description='collection of tasks that produce the tables to merge')

    def requires(self):
        return self.tables


class TableFuncApplier(TableTransform):
    func = luigi.Parameter(
        description='function to perform transformation on table')

    def apply_func(self, table):
        if self.multiple_columns:
            othercols = self.cols[1:]
            if len(othercols) == 1:  # extract single additional column name
                othercols = othercols[0]
            self.func(table, self.colname, othercols)
        else:
            self.func(table, self.colname)

