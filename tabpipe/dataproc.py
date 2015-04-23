"""
Pandas/luigi-based tabular data processing framework.

Pandas provides powerful utilities for working with tabular data in Python. It
sits upon numpy, which enables efficient storage and computation. Luigi is a
data pipelining framework that allows complex dependency resolution between data
processing tasks. Dependencies are handled by producing intermediate files. A
dependent process searches for its dependencies' output files to determine if
its dependencies are met. If the files are not found, the tasks that produce
them are run. This continues recursively until all dependencies have been met
and the current task can be run. Tasks which do not depend on each other can be
run in parallel. Luigi also facilitates simple distribution on Hadoop clusters
and across HDFS files.

Now to the framework: tabpipe

The main unit of work is the Preprocessor. There are 5 subordinate tasks
involved in each preprocessing task:

    1.  Cleaner
    2.  FeatureProducer
    3.  Mapper
    4.  Merger
    5.  Writer

Cleaning involves inferring missing values in certain columns based on other
values in that column or related columns. It also involves discarding rows which
cannot be used based on some criteria which defines usability.

Feature production involves producing new columns from existing columns.

Mapping operates in the sense that a function operates. Each value in a column
is input into a mapping function that deterministically produces a substitute
value. This is often useful for changing set categorical values into numerical
id encodings or ordinal encodings.

Merging involves the combination of several different tables into one table.

In reality, these processes may take place in different order and be repeated.
Currently, this framework assumes each subunit takes place sequentially, in the
order listed above. Future refactorings may introduce the ability to, for
instance, have something like this:

    cleaning -> feature production -> merging -> feature production ->
    mapping -> merging -> writing

"""
import os
import hashlib

import pandas as pd
import luigi



def parse_colnames(thing):
    """Can be str, csv string, or tuple of str."""
    if hasattr(thing, '__len__'):
        return thing
    elif ',' in thing:
        return thing.split(',')
    elif isinstance(thing, basestring):
        return thing
    else:
        raise ValueError('invalid primary id: {}'.format(thing))


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
    colname = luigi.Parameter(
        description='name of column to scrub')
    # Not used for some subclasses; may reconsider having this here.
    othercols = luigi.Parameter(
        default=[],
        description='other columns to use for scrubbing')
    outname = luigi.Parameter(
        description='filename to output transformed table to')

    @property
    def usecols(self):
        cols = [0, self.colname]  # include 0 for index
        if self.othercols:
            if hasattr(self.othercols, '__iter__'):
                cols.extend(self.othercols)
            else:  # string
                cols.append(self.othercols)
        return cols

    def read_output_table(self):
        with self.output().open() as f:
            # Only index and `colname` are written to output.
            return pd.read_csv(f, index_col=0, usecols=(0, self.colname,))

    def read_input_table(self):
        with self.input().open() as f:
            return pd.read_csv(f, index_col=0, usecols=self.usecols)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.savedir, self.outname))

    def run(self):
        raise NotImplementedError('Base class does not implement run method.')


class TableTransform(TableBase):
    """Takes a task that outputs a table, transforms the table, writes it."""
    table = luigi.Parameter(
        description='task that outputs table to transform')

    def requires(self):
        return self.table


class MultipleTableTransform(TableBase):
    """Takes several tables and outputs one table."""
    tables = luigi.Parameter(
        description='list of tasks that produce the tables to merge')

    def requires(self):
        return self.tables


class TableMerger(MultipleTableTransform):
    """Merge multiple DataFrames together."""
    on = luigi.Parameter(
        default=None,
        description='name of column (or csv string of names) to merge on')

    def run(self):
        """Merge all input data frames on the given colname(s)."""
        on = parse_colnames(self.on)


class ColumnScrubber(TableTransform):
    """Infer missing values or clean existing values in a particular column."""
    func = luigi.Parameter(
        description='scrubbing function; operate on table; change values in-place')
    outname = luigi.Parameter(
        default=None,
        description='table output filename; default is `table`-`colname`-fill')

    def output(self):
        if self.outname:
            path = os.path.join(self.savedir, self.outname)
        else:
            outname = '-'.join((self.table.tname, self.colname, 'fill'))
            path = os.path.join(self.savedir, outname)
        return luigi.LocalTarget(path)

    def run(self):
        """Load only `colname` + `othercols` and fill missing values using
        `func`. Write the column for `colname` to `outname` with the index.
        We assume the input table has an index.
        """
        table = self.read_input_table()

        # May or may not have additional columns; functions shouldn't need to be
        # written to take them, so we multiplex it here.
        if self.othercols:
            self.func(table, self.colname, self.othercols)
        else:
            self.func(table, self.colname)

        with self.output().open('w') as f:
            table.to_csv(f, columns=(self.colname,), index=True)


class ColumnReplacer(TableTransform):
    """Replace one column in a table with another from another table."""
    replacement = luigi.Parameter(
        description='task that outputs the table with the replacement column')
    outname = luigi.Parameter(
        default=None,
        description='table output filename; default is table/replacement name hash')

    def requires(self):
        return {'table': self.table,
                'replacement': self.replacement}

    @property
    def usecols(self):
        return None  # use all columns

    def read_input_table(self):
        with self.input()['table'].open() as f:
            return pd.read_csv(f, index_col=0, usecols=self.usecols)

    def output(self):
        if self.outname:
            return luigi.LocalTarget(self.outname)
        else:
            hashes = ''.join((self.table.thash, self.replacement.thash))
            outname = hashlib.sha1(hashes).hexdigest()
            return luigi.LocalTarget(outname)

    def run(self):
        """Load `colname` from the `replacement` table and replace the column
        with `colname` in `table`.
        """
        inputs = self.input()
        with inputs['table'].open() as f:
            df = pd.read_csv(f, index_col=0)
        with inputs['replacement'].open() as f:
            col = pd.read_csv(f, index_col=0)

        # Replace column.
        df[self.colname] = col[self.colname]
        with self.output().open('w') as f:
            df.to_csv(f, index=True)


class RowFilterer(TableTransform):
    """Filter the rows of a table; output indices to remove."""
    func = luigi.Parameter(
        description='filtering function; returns indices of rows to remove')
    colnames = luigi.Parameter(
        description='columns to be used as filtering criteria')

    def run(self):
        """Load only `colnames` from `fname`. Output indices of rows to be
        removed.
        """
        pass


class RowRemover(TableTransform):
    """Remove the indices from one table specified by another."""
    filter_tables = luigi.Parameter(
        description='task that outputs table(s) with indices to remove')
    on = luigi.Parameter(
        description='index or indices to join `table` and `filter_tables` on')

    def requires(self):
        return {'table': self.table,
                'filter_tables': self.filter_tables}

    def run(self):
        """Load `toremove` table and remove specified rows from `table`."""
        pass


class Cleaner(TableTransform):
    """Fill rows, replace columns, filter rows, remove rows, output."""
    scrubbers = luigi.Parameter(
        description='list of dict: `colname`, `othercols`, `func` (see ColumnScrubber)')
    filterers = luigi.Parameter(
        description='list of dict: `name`, `colnames`, `func` (see RowFilterer)')
    primary_id = luigi.Parameter(
        description='colname of primary id, or csv string of colnames, or tuple')


    def __init__(self, *args, **kwargs):
        super(Cleaner, self).__init__(*args, **kwargs)

        # Parse out primary id.
        primary_id = parse_colnames(self.primary_id)

        # Create scrubbers and replacers.
        self.filler_tasks = []
        self.replacer_tasks = []
        table_task = self.table
        for spec in self.scrubbers:
            # Output name is combo of original table name, colname being
            # being filled, and the word fill.
            spec['table'] = self.table
            filler_task = ColumnScrubber(**spec)

            # Replacer tasks operate in a pipeline fashion; the first one
            # takes the original table, the second takes the table output from
            # the last one, and so on until the final column replacement
            # produces the table that gets passed into the row filterers.
            replacer_task = ColumnReplacer(
                table=table_task, replacement=filler_task,
                colname=spec['colname'])
            table_task = replacer_task

            self.filler_tasks.append(filler_task)
            self.replacer_tasks.append(replacer_task)

        self.row_filterers = []
        for spec in self.filterers:
            # Output name is combo of last column replacer output name and
            # the filter criteria name.
            parts = (replacer_task.table_name, spec['name'])
            spec['outname'] = '-'.join(parts)
            spec['table'] = replacer_task
            self.row_filterers.append(RowFilterer(**spec))

        # The remover is the final step in the Cleaner pipeline. All indices
        # marked for removal are combined and those rows are removed from the
        # table from the last column replacer. This is the final output.
        self.remover = RowRemover(
            table=replacer_task, filter_tables=self.row_filterers,
            outname=self.outname, on=primary_id)

    def run(self):
        if not self.remover.complete():
            self.remover.run()


class FeatureEngineer(TableTransform):
    pass

class FeatureProducer(TableTransform):
    pass

class ColumnMapper(TableTransform):
    pass

class ValueSubstituter(TableTransform):
    pass

class Mapper(TableTransform):
    pass

class Preprocessor(TableTransform):
    pass


if __name__ == "__main__":
    luigi.run()
