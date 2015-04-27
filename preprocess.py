import os
import sys

import pandas as pd
import numpy as np

from writer import write_libfm


# What attributes do we want to write?
finstructor = ['iid', 'iclass', 'irank', 'itenure']
fcourse = ['cid', 'clevel', 'cdisc', 'termnum']
fstudent = ['sid', 'cohort', 'race', 'sex', 'zip', 'major', 'hs']
CVALS = finstructor + fcourse + fstudent
RVALS = ['age', 'hsgpa', 'sat', 'chrs']


# Alphabetical grade to quality points
# Guides consulted:
# https://www.gmu.edu/academics/catalog/0203/apolicies/examsgrades.html
# http://catalog.gmu.edu/content.php?catoid=15&navoid=1168
# https://registrar.gmu.edu/topics/special-grades/
grade2pts = {
    'A+':   4.0,
    'A':    4.0,
    'A-':   3.67,
    'B+':   3.33,
    'B':    3.00,
    'B-':   2.67,
    'C+':   2.33,
    'C':    2.00,
    'C-':   1.67,
    'D':    1.00,
    'F':    0.00,
    'IN':   0.00,    # Incomplete
    'S':    np.nan,    # Satisfactory (passing; C and up, no effect on GPA)
    'NC':   np.nan,    # No Credit (often C- and below)
    'W':    np.nan,    # Withdrawal (does not affect grade)
    'NR':   np.nan,  # Not Reported (possibly honor code violation)
    'AU':   np.nan,  # Audit
    'REG':  np.nan,  # ?
    'IX':   np.nan,  # Incomplete Extension
    'IP':   np.nan,  # In Progress
    'nan':  np.nan,  # Unknown
    np.nan: np.nan   # Unknown (for iteration over Series)
}


# Fix up the grades where possible.
def fill_grdpts(series):
    """Fill in missing values for grade quality points. This also replaces any
    existing grdpts values with the propper mapping from the letter grade. We
    assume the letter grades are more reliable when they are present.
    """
    # TODO: can fill missing lab grades with lecture grades if we can match them
    if series['GRADE'] != np.nan:
        return grade2pts[series['GRADE']]
    else:
        return series['grdpts']


def map_columns(df, colnames, newname=None, remove=True):
    """Map set categorical column(s) to contiguous numerical values.
    Write the mapping to a csv file and replace the column values in place in
    the data frame passed in. Note that nan values are left as-is; they are not
    mapped to numerical values. Hence the type of the resulting column will be
    float64 if nan values are present.

    :param colnames: One or more colnames to convert.
    :type colnames: str or iterable of str.
    :param str newname: New column name to use.
    :param bool remove: If True, remove old columns after mapping.
    """
    multicol = hasattr(colnames, '__iter__')
    if newname is None:
        if multicol:
            raise ValueError(
                "Must pass replacement value name when mapping 2+ cols")
        else:
            newname = colnames  # direct replacement

    # We'd like to replace the old column name by doing a merge on the old
    # names. However, this will fail if the new name is the same as the old one,
    # or one of the old ones. Let's check for this.
    same_name = True if newname in df.columns else False

    # Create the mapping and save it.
    colnames = list(colnames) if multicol else [colnames]
    outname = '_'.join(['-'.join(colnames), 'idmap.csv'])
    outpath = os.path.join('idmap', outname)
    idmap = df[colnames].drop_duplicates().sort(colnames)[colnames]

    # Remove nan mapping and reset index; this prevents mapping of nan values to
    # numerical indices, which would likely be nonsensical.
    idmap = idmap[~idmap.isnull().any(axis=1)].reset_index()[colnames]
    idmap.to_csv(outpath, index_label='index')

    # Now replace the values. If we're replacing with the same name, we need to
    # make a temporary column that won't be mangled during the merge.
    if not same_name:
        idmap[newname] = idmap.index
    else:
        newid = 'idx_%s' % newname
        idmap[newid] = idmap.index

    # Ensure data types match up; may be unnecessary when not writing first.
    for colname in colnames:
        dtype = df[colname].dtype.type
        idmap[colname] = idmap[colname].values.astype(dtype)

    # Add the new id-mapped column to the original DataFrame by merging on the
    # columns the mapping was produced from.
    out = df.merge(idmap, how='left', on=colnames)
    if same_name:
        out[newname] = out[newid]
        del out[newid]

    # Drop mapped columns if requested
    if remove:
        if same_name:
            colnames.remove(newname)
        for name in colnames:
            del out[name]

    return out


def read_idmap(colnames):
    """Read the idmap produced while mapping the given colnames."""
    multicol = hasattr(colnames, '__iter__')
    colnames = list(colnames) if multicol else [colnames]
    outname = '_'.join(['-'.join(colnames), 'idmap.csv'])
    outpath = os.path.join('idmap', outname)
    with open(outpath) as f:
        return pd.read_csv(f, index_col=0)


def extract_clevel(cnum):
    """Extract the course level from the course number."""
    if cnum == np.nan:
        return np.nan
    cnum = str(cnum).strip()
    if not cnum:
        return np.nan
    digits = filter(lambda c: c.isdigit(), cnum)
    if not digits:
        return np.nan
    return digits[0]


def preprocess(outname='preprocessed-data.csv'):
    demog = pd.read_csv('data/nsf_demographics.csv')

    # Specify which columns to keep and read in data.
    courses_cols = ['id', 'TERMBNR', 'DISC', 'CNUM', 'GRADE', 'HRS',
                    'grdpts', 'INSTR_LNAME', 'INSTR_FNAME', 'class',
                    'instr_rank', 'instr_tenure', 'TITLE']
    courses = pd.read_csv('data/nsf_courses.csv', usecols=courses_cols)

    students_cols = ['id', 'cohort', 'TERMBNR', 'PMAJR', 'termgpa',
                     'term_earn_hrs', 'cumgpa']
    students = pd.read_csv('data/nsf_student.csv', usecols=students_cols)

    admiss_cols = ['id', 'cohort', 'Permanent_Address_ZIP', 'HSGPA',
                   'SAT_Total_1600', 'HS_CEEB_Code']
    admiss = pd.read_csv('data/nsf_admissions.csv', usecols=admiss_cols)

    # Merge student and demographic data.
    out = students.merge(demog, how='left', on='id')

    # Map several of the ids in this table.
    out = map_columns(out, 'race', 'race')
    out = map_columns(out, 'SEX', 'sex')

    # # Merge courses and student data.
    out = courses.merge(out, how='left', on=('id', 'TERMBNR'))

    # Map several of the ids in the resulting table.
    out = map_columns(out, 'TERMBNR', 'termnum')
    out = map_columns(out, ['DISC', 'CNUM', 'HRS'], 'cid', remove=False)
    out = map_columns(out, 'DISC', 'cdisc')
    out = map_columns(out, ['INSTR_LNAME', 'INSTR_FNAME'], 'iid')
    out = map_columns(out, 'class', 'iclass')
    out = map_columns(out, 'instr_rank', 'irank')
    out = map_columns(out, 'instr_tenure', 'itenure')
    out = map_columns(out, 'PMAJR', 'major')

    # Get course level from CNUM.
    out['clevel'] = out['CNUM'].apply(extract_clevel)
    del out['CNUM']

    # Fill in missing values for quality points.
    out['grdpts'] = out.apply(fill_grdpts, axis=1)
    del out['GRADE']

    # Next, merge with the admissions data.
    out = out.merge(admiss, how='left', on=('id', 'cohort'))

    # Map columns.
    out = map_columns(out, 'Permanent_Address_ZIP', 'zip')
    out = map_columns(out, 'HS_CEEB_Code', 'hs')
    out = map_columns(out, 'id', 'sid')

    # Map cohort using TERMBNR mapping.
    idmap = read_idmap('TERMBNR')
    newname = 'new_cohort'
    idmap[newname] = idmap.index  # tmp column; avoid conflict on cohort
    out = out.merge(idmap, how='left', left_on='cohort', right_on='TERMBNR')
    out['cohort'] = out[newname]  # replace old with tmp
    del out[newname]    # remove tmp column
    del out['TERMBNR']  # remove extra TERMBNR column acquired during merge

    # Rename some columns.
    out.rename(columns={
        'HSGPA': 'hsgpa',
        'SAT_Total_1600': 'sat',
        'HRS': 'chrs',
        'ENTRY_AGE': 'age'
    }, inplace=True)

    # Remove any records without grdpts
    out = out[~out['grdpts'].isnull()]

    # Drop duplicate (student, course) pairs, keeping last attempt.
    out = out.sort(['termnum', 'sid'])
    out = out.drop_duplicates(['sid', 'cid'], take_last=True)

    # Write out the data to a csv file.
    out.to_csv(outname, index=False)


# Do everything in main to make IPython inspection simple.
if __name__ == "__main__":
    preprocess('data/preprocessed-data.csv')
