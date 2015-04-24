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
import numpy as np
import luigi
import luigi.worker


def schedule_task(task, verbose=False):
    if verbose:
        luigi.interface.setup_interface_logging()
    sch = luigi.scheduler.CentralPlannerScheduler()
    w = luigi.worker.Worker(scheduler=sch)
    w.add(task)
    w.run()


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


class ColumnScrubber(TableFuncApplier):
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
        self.apply_func(table)
        with self.output().open('w') as f:
            table.to_csv(f, columns=(self.colname,), index=True)


class ColumnReplacer(TableTransform):
    """Replace one column in a table with another from another table."""
    replacement = luigi.Parameter(
        description='task that outputs the table with the replacement column')
    outname = luigi.Parameter(
        default=None,
        description='table output filename; default is table/replacement name hash')
    usecols = None  # read all columns from input table

    def requires(self):
        return {'table': self.table,
                'replacement': self.replacement}

    def output(self):
        """Either the outname passed or a sha1 hash of a hash of the source
        table name and a hash of the replacement table name.
        """
        if self.outname:
            path = os.path.join(self.savedir, self.outname)
        else:
            hashes = ''.join((self.table.thash, self.replacement.thash))
            outname = hashlib.sha1(hashes).hexdigest()
            path = os.path.join(self.savedir, outname)
        return luigi.LocalTarget(path)

    def run(self):
        """Load `colname` from the `replacement` table and replace the column
        with `colname` in `table`. The output includes the entire original table
        with the single column replaced.
        """
        inputs = self.input()
        df = self.read_input_table()
        with inputs['replacement'].open() as f:
            col = pd.read_csv(f, index_col=0, usecols=(0, self.colname))

        # Replace column.
        df[self.colname] = col[self.colname]
        with self.output().open('w') as f:
            df.to_csv(f, index=True)


class RowFilterer(TableTransform):
    """Filter the rows of a table; output indices to remove."""
    func = luigi.Parameter(
        description='filtering function; returns indices of rows to remove')
    name = luigi.Parameter(
        description='unique name of the filtering criteria; used for output')
    outname = luigi.Parameter(
        default=None,
        description='output filename; default: input + hash of colnames + filter')

    def output(self):
        """Either the outname passed or a combination of the input table name,
        the name of the filtering criterion, and a hash of the colnames:
            tname-criterionName-hash.
        """
        if self.outname:
            path = os.path.join(self.savedir, self.outname)
        else:
            hash = hashlib.sha1(''.join(self.cols)).hexdigest()
            outname = '-'.join([self.table.tname, hash, self.name])
            path = os.path.join(self.savedir, outname)
        return luigi.LocalTarget(path)

    def run(self):
        """Load only `colnames` from `fname`. Output indices of rows to be
        removed.
        """
        df = self.read_input_table()
        indices = self.func(df, self.cols)
        data = pd.DataFrame({'indices': indices.values})

        with self.output().open('w') as f:
            data.to_csv(f, index=False)


class RowRemover(TableTransform):
    """Remove the indices from one table specified by another."""
    filter_tables = luigi.Parameter(
        description='task that outputs table(s) with indices to remove')
    on = luigi.Parameter(
        default=None,
        description='colname for indices; default is index')
    colnames = None  # no column names necessary
    usecols = None  # read all columns from input table
    outname = luigi.Parameter(
        default=None,
        description='output filename; default: input-<hash `filter_tables`>')

    def requires(self):
        return {'table': self.table,
                'filter_tables': self.filter_tables}

    def read_indices(self):
        indices = []
        for tfile in self.input()['filter_tables']:
            with tfile.open() as f:
                # We expect to get an empty data frame; we want its index.
                idx_table = pd.read_csv(f, index_col=0)
                indices.append(idx_table.index.values)

        # Combine all indices to remove and get unique indices.
        return np.unique(np.concatenate(indices))

    @property
    def indices_to_remove(self):
        """Cache indices to remove; return from cache."""
        try:
            return self._toremove
        except AttributeError:
            self._toremove = self.read_indices()
            return self._toremove

    def output(self):
        """Either the outname passed or a combination of the input table name
        and a hash of the filter table names. We would really like to use the
        indices being removed. Different filtering criteria may actually end up
        flagging the same indices for removal. This way, the task won't be rerun
        in that case. However, the complete method needs to know the name of the
        output file to see if it should generate those very same input files
        that would contain the indices, so we can't use that method.
        """
        if self.outname:
            path = os.path.join(self.savedir, self.outname)
        else:
            inputs = [f.path for f in self.input()['filter_tables']]
            hash = hashlib.sha1(''.join(inputs)).hexdigest()
            outname = '-'.join((self.table.tname, hash))
            path = os.path.join(self.savedir, outname)
        return luigi.LocalTarget(path)

    def run(self):
        """Load input table, load all indices from `filter_tables`, and combine
        them, then remove all rows with those index values in the `on` column.
        """
        df = self.read_input_table()
        col = df[self.on] if self.on else df.index
        df = df[~col.isin(self.indices_to_remove)]
        with self.output().open('w') as f:
            df.to_csv(f, index=True)


class Cleaner(TableTransform):
    """Fill rows, replace columns, filter rows, remove rows, output."""
    scrubbers = luigi.Parameter(
        default=[],  # may not need any scrubbers
        description='list of dict: `colnames`, `func` (see ColumnScrubber)')
    filterers = luigi.Parameter(
        default=[],  # may not need any filterers
        description='list of dict: `name`, `colnames`, `func` (see RowFilterer)')
    primary_id = luigi.Parameter(
        description='colname of primary id, or csv string of colnames, or tuple')
    colnames = None

    def __init__(self, *args, **kwargs):
        super(Cleaner, self).__init__(*args, **kwargs)

        # Parse out primary id.
        primary_id = parse_colnames(self.primary_id)

        # Create scrubbers and replacers.
        self.scrubber_tasks = []
        self.replacer_tasks = []
        table_task = self.table
        for spec in self.scrubbers:
            # Output name is combo of original table name, colname being
            # being filled, and the word fill.
            spec['table'] = self.table
            scrubber_task = ColumnScrubber(**spec)

            # Replacer tasks operate in a pipeline fashion; the first one
            # takes the original table, the second takes the table output from
            # the last one, and so on until the final column replacement
            # produces the table that gets passed into the row filterers.
            replacer_task = ColumnReplacer(
                table=table_task, replacement=scrubber_task,
                colnames=spec['colnames'])
            table_task = replacer_task

            self.scrubber_tasks.append(scrubber_task)
            self.replacer_tasks.append(replacer_task)

        # If we have no scrubber tasks, the input dtable is still the table
        # we're working with.
        table_task = replacer_task if self.scrubbers else self.table

        self.row_filterers = []
        for spec in self.filterers:
            # Output name is combo of last column replacer output name and
            # the filter criteria name.
            parts = (table_task.tname, spec['name'])
            spec['outname'] = '-'.join(parts)
            spec['table'] = replacer_task
            self.row_filterers.append(RowFilterer(**spec))

        # If we have no filtering tasks, the current dtable is the table from
        # the last replacer, otherwise it's the table from the remover.
        if self.filterers:
            # All indices marked for removal are combined and those rows are
            # removed from the table from the last column replacer. This is the
            # final output.
            self.final_task = RowRemover(
                table=replacer_task, filter_tables=self.row_filterers,
                outname=self.outname, on=primary_id)
        else:
            self.final_task = table_task

    def delete_intermediates(self):
        """Delete all intermediate output files."""
        all_tasks = \
            self.scrubber_tasks + self.replacer_tasks + self.row_filterers
        if self.filterers:
            all_tasks.append(self.final_task)

        for task in all_tasks:
            try:
                os.remove(task.output().path)
            except OSError:
                pass

    def run(self):
        schedule_task(self.final_task, verbose=False)


class TableMerger(MultipleTableTransform):
    """Merge multiple DataFrames together."""
    on = luigi.Parameter(
        default=None,
        description='name of column (or csv string of names) to merge on')

    def run(self):
        """Merge all input data frames on the given colname(s)."""
        on = parse_colnames(self.on)


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
