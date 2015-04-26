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
    'S':    3.00,    # Satisfactory (passing; C and up, no effect on GPA)
    'NC':   1.00,    # No Credit (often C- and below)
    'W':    1.00,    # Withdrawal (does not affect grade)
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
    """Fill in missing values for grade quality points."""
    # TODO: can fill missing lab grades with lecture grades if we can match them
    return (grade2pts[series['GRADE']] if series['GRADE'] != np.nan
            else series['grdpts'])


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


def preprocess(outname='preprocessed-data.csv'):
    courses = pd.read_csv('data/nsf_courses.csv')
    students = pd.read_csv('data/nsf_student.csv')
    admiss = pd.read_csv('data/nsf_admissions.csv')
    demog = pd.read_csv('data/nsf_demographics.csv')

    # Filter out unneeded columns from the tables.
    courses_cols = ['id', 'TERMBNR', 'DISC', 'CNUM', 'GRADE', 'HRS',
                    'grdpts', 'INSTR_LNAME', 'INSTR_FNAME', 'class',
                    'instr_rank', 'instr_tenure']
    courses = courses[courses_cols]

    students_cols = ['id', 'cohort', 'TERMBNR', 'PMAJR', 'termgpa',
                     'term_earn_hrs', 'cumgpa']
    students = students[students_cols]

    admiss_cols = ['id', 'cohort', 'Permanent_Address_ZIP', 'HSGPA',
                   'SAT_Total_1600', 'HS_CEEB_Code']
    admiss = admiss[admiss_cols]

    # Merge student and demographic data.
    out = students.merge(demog, how='outer', on=('id',))

    # Map several of the ids in this table.
    out = map_columns(out, 'ENTRY_AGE', 'age')
    out = map_columns(out, 'race', 'race')
    out = map_columns(out, 'SEX', 'sex')

    # # Merge students and courses data.
    out = out.merge(courses, how='outer', on=('id', 'TERMBNR'))

    # Map several of the ids in the resulting table.
    out = map_columns(out, 'TERMBNR', 'termnum')
    out = map_columns(out, ['CNUM', 'DISC', 'HRS'], 'cid', remove=False)
    out = map_columns(out, 'DISC', 'cdisc')
    out = map_columns(out, ['INSTR_LNAME', 'INSTR_FNAME'], 'iid')
    out = map_columns(out, 'class', 'iclass')
    out = map_columns(out, 'instr_rank', 'irank')
    out = map_columns(out, 'instr_tenure', 'itenure')
    out = map_columns(out, 'PMAJR', 'major')

    # Get course level from CNUM.
    out['clevel'] = out['CNUM'].apply(
        lambda cnum: str(cnum)[0] if cnum != np.nan else np.nan)
    del out['CNUM']

    # Fill in missing values for quality points.
    out['grdpts'] = out.apply(fill_grdpts, axis=1)
    del out['GRADE']

    # Next, merge with the admissions data.
    out = out.merge(admiss, how='outer', on=('id', 'cohort'))

    # Map columns.
    out = map_columns(out, 'HS_CEEB_Code', 'hs')
    out = map_columns(out, 'cohort', 'cohort')
    out = map_columns(out, 'id', 'sid')
    out = map_columns(out, 'Permanent_Address_ZIP', 'zip')

    # Rename some columns.
    out.rename(columns={
        'HSGPA': 'hsgpa',
        'SAT_Total_1600': 'sat',
        'HRS': 'chrs'
    }, inplace=True)

    # Drop definite duplicates.
    out = out.sort(['sid', 'termnum'])
    out = out.drop_duplicates(['sid', 'cid', 'iid', 'grdpts'], take_last=True)

    # Remove any records without grdpts
    out = out[~out['grdpts'].isnull()]

    # Write out the data to a csv file.
    out.to_csv(outname, index=False)



# Do everything in main to make IPython inspection simple.
if __name__ == "__main__":
    courses = pd.read_csv('data/nsf_courses.csv')
    students = pd.read_csv('data/nsf_student.csv')
    admiss = pd.read_csv('data/nsf_admissions.csv')
    demog = pd.read_csv('data/nsf_demographics.csv')

    # Filter out unneeded columns from the tables.
    courses_cols = ['id', 'TERMBNR', 'DISC', 'CNUM', 'GRADE', 'HRS',
                    'grdpts', 'INSTR_LNAME', 'INSTR_FNAME', 'class',
                    'instr_rank', 'instr_tenure']
    courses = courses[courses_cols]

    students_cols = ['id', 'cohort', 'TERMBNR', 'PMAJR', 'termgpa',
                     'term_earn_hrs', 'cumgpa']
    students = students[students_cols]

    admiss_cols = ['id', 'cohort', 'Permanent_Address_ZIP', 'HSGPA',
                   'SAT_Total_1600', 'HS_CEEB_Code']
    admiss = admiss[admiss_cols]

    # Merge student and demographic data.
    out = students.merge(demog, how='outer', on=('id',))

    # Map several of the ids in this table.
    out = map_columns(out, 'ENTRY_AGE', 'age')
    out = map_columns(out, 'race', 'race')
    out = map_columns(out, 'SEX', 'sex')

    # # Merge students and courses data.
    out = out.merge(courses, how='outer', on=('id', 'TERMBNR'))

    # Map several of the ids in the resulting table.
    out = map_columns(out, ['CNUM', 'DISC', 'HRS'], 'cid', remove=False)
    out = map_columns(out, 'DISC', 'cdisc')
    out = map_columns(out, ['INSTR_LNAME', 'INSTR_FNAME'], 'iid')
    out = map_columns(out, 'class', 'iclass')
    out = map_columns(out, 'instr_rank', 'irank')
    out = map_columns(out, 'instr_tenure', 'itenure')
    out = map_columns(out, 'PMAJR', 'major')

    # Get course level from CNUM.
    out['clevel'] = out['CNUM'].apply(
        lambda cnum: str(cnum)[0] if cnum != np.nan else np.nan)
    del out['CNUM']

    # Fill in missing values for quality points.
    out['grdpts'] = out.apply(fill_grdpts, axis=1)
    del out['GRADE']

    # Next, merge with the admissions data.
    out = out.merge(admiss, how='outer', on=('id', 'cohort'))

    # Map columns.
    out = map_columns(out, 'HS_CEEB_Code', 'hs')
    out = map_columns(out, 'id', 'sid')
    out = map_columns(out, 'Permanent_Address_ZIP', 'zip')

    # Replace cohort using TERMBNR mapping
    out = map_columns(out, 'TERMBNR', 'termnum')
    idmap = read_idmap('TERMBNR')
    newname = 'new_cohort'
    idmap[newname] = idmap.index
    out = out.merge(idmap, how='left', left_on='cohort', right_on='TERMBNR')
    out['cohort'] = out[newname]
    del out[newname]
    del out['TERMBNR']

    # Rename some columns.
    out.rename(columns={
        'HSGPA': 'hsgpa',
        'SAT_Total_1600': 'sat',
        'HRS': 'chrs'
    }, inplace=True)

    # Drop definite duplicates.
    out = out.sort(['sid', 'termnum'])
    out = out.drop_duplicates(['sid', 'cid', 'iid', 'grdpts'], take_last=True)

    # Remove any records without grdpts
    out = out[~out['grdpts'].isnull()]

    # What attributes do we want to write?
    finstructor = ['iid', 'iclass', 'irank', 'itenure']
    fcourse = ['cid', 'clevel', 'cdisc', 'termnum']
    fstudent = ['sid', 'cohort', 'race', 'sex', 'zip', 'major', 'hs']
    cvals = finstructor + fcourse + fstudent
    rvals = ['age', 'hsgpa', 'sat', 'chrs']

    # Write out the data to a csv file.
    out.to_csv('preprocessed-ers-data.csv', index=False)
    # with open('preprocessed-ers-data.libfm', 'w') as f:
    #     write_libfm(f, out, target='grdpts', cvals=cvals, rvals=rvals)
