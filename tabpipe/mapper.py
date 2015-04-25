"""
Implement the Mapper subunit.

"""
import os

import luigi
import pandas as pd

from util import *
from core import TableTransform, ColumnReplacer


class ColumnIdMapper(TableTransform):
    """Produce contiguous ids from 1+ columns; one for each unique value."""
    newid_name = luigi.Parameter(
        description='colname to give to new id-mapped column')
    outname = luigi.Parameter(
        default=None,
        description='output filename; defualt: input-idmap_<cnames>')

    def output(self):
        if self.outname:
            path = os.path.join(self.savedir, self.outname)
        else:
            colname_abbrev = abbrev_names(self.cols)
            outname = '-'.join([self.table.tname, 'idmap_%s' % colname_abbrev])
            path = os.path.join(self.savedir, outname)
        return luigi.LocalTarget(path)

    def run(self):
        """Produce contiguous id map; if nan values are present, they get their
        own unique id as well.
        """
        df = self.read_input_table()
        mapping = df[self.cols].drop_duplicates().reset_index()[self.cols]
        with self.output().open('w') as f:
            mapping.to_csv(f, index=True, index_label='index')


# WARNING: this will change the index
class ValueSubber(TableTransform):
    """Substitute existing column values for new ones using a 1:1 map."""
    idmap = luigi.Parameter(
        description='task that outputs idmap table')
    outname = luigi.Parameter(
        default=None,
        description='output filename; defualt: input-Sub_<cnames>')
    usecols = None  # read all columns from input table.

    def requires(self):
        return {
            'table': self.table,
            'idmap': self.idmap
        }

    def output(self):
        if self.outname:
            path = os.path.join(self.savedir, self.outname)
        else:
            colname_abbrev = abbrev_names(self.cols)
            outname = '-'.join([self.table.tname, 'Sub_%s' % colname_abbrev])
            path = os.path.join(self.savedir, outname)
        return luigi.LocalTarget(path)

    def run(self):
        """Replace all values in given colname using an idmap."""
        with self.input()['idmap'].open() as f:
            idmap = pd.read_csv(f, index_col=0)

        # We'll create the mapping by inserting duplicating the index as a
        # column with the new column name, then merging on the columns that were
        # mapped to create the index. Then we'll remove the old columns, since
        # we no longer need them.
        newid = self.idmap.newid_name
        same_name = False
        if newid in idmap.columns:  # same name replacement here!
            same_name = True
            oldid = 'old_%s' % newid
            idmap[oldid] = idmap.index
        else:
            idmap[newid] = idmap.index

        # Determine which columns to keep
        df = self.read_input_table()
        cols = list(df.columns.values) + [newid]
        cols = [colname for colname in cols if colname not in self.cols]
        if same_name:  # new id name will have been removed
            cols.append(newid)
            cols.append(oldid)

        # Ensure data types remain consistent.
        for colname in self.cols:
            dtype = df[colname].dtype.type
            idmap[colname] = idmap[colname].values.astype(dtype)

        # merge with idmap to add new id mapping; retain current index and
        # ordering.
        merged = df.reset_index()\
                   .merge(idmap, how='left', on=self.cols)\
                   .set_index('index').sort_index()[cols]
        merged.index.name = None  # get rid of index name

        if same_name:
            merged[newid] = merged[oldid]
            cols.remove(oldid)

        with self.output().open('w') as f:
            merged.to_csv(f, columns=cols, index=True)


class Mapper(TableTransform):
    """Replace 1+ set categorical columns with unique numerical id mappings."""
    maps = luigi.Parameter(
        description='dict: `newid`, `colnames`: list of columns to map')
    outname = luigi.Parameter(
        default=None,
        description='output filename; default: input-Map<colnames-abbreviated>')
    colnames = None  # TODO: better way of dealing with this; still have props

    @property
    def abbrev(self):
        return '-'.join(['_'.join([newid, abbrev_names(cols)])
                         for newid, cols in self.maps.items()])

    @property
    def outpath(self):
        if self.outname:
            return os.path.join(self.savedir, self.outname)
        else:
            outname = '-'.join([self.table.tname, 'Map--%s' % self.abbrev])
            return os.path.join(self.savedir, outname)

    def output(self):
        """Abbreviate colnames by attempting to take as few of the first few
        letters as necessary to get unique namings. Smash these all together and
        use title casing. So for instance: colnames=('grade', 'gpa', 'rank')
        would produce: Map_GrGpRa.
        """
        return luigi.LocalTarget(self.outpath)

    def __init__(self, *args, **kwargs):
        """Each column marked for mapping is mapped as follows:

        1.  An id mapping is generated by a ColumnIdMapper
        2.  A ValueSubber reads the original table and the newly created idmap,
            substitues all values using the idmap, and discards the columns used
            to make the idmap. This table is written to the output file.

        The final task is the last ValueSubber

        """
        super(Mapper, self).__init__(*args, **kwargs)

        self.idmaps = {}
        self.mapper_tasks = []
        self.subber_tasks = []

        table_task = self.table
        for newid, colnames in self.maps.items():
            idmapper = ColumnIdMapper(
                table=self.table, newid_name=newid, colnames=colnames)
            self.idmaps[newid] = idmapper.output().path

            subber = ValueSubber(
                table=table_task, idmap=idmapper, colnames=colnames,
                outname=self.outpath)
            table_task = subber

            self.mapper_tasks.append(idmapper)
            self.subber_tasks.append(subber)

        self.final_task = subber

    def run(self):
        schedule_task(self.final_task, verbose=False)

    @property
    def all_tasks(self):
        return self.mapper_tasks + self.subber_tasks

    def delete_intermediates(self):
        """Delete all intermediate output files."""
        for task in self.all_tasks:
            try:
                os.remove(task.output().path)
            except OSError:
                pass
