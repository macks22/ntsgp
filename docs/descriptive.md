% DegreePlanner Data Exploration
% Mack Sweeney

# Purpose

This document provides a descriptive overview of the data which will be used to
construct the DegreePlanner (DP) system. It provides descriptive statistics,
visualizations, and textual summaries of salient data characteristics.

# Entities in the Data

There are a total of 4 files which currently hold the entirety of the data
available. These are:

1.  admissions.csv
2.  courses.csv
3.  degrees.csv
4.  student.csv

There are three main entities of interest in this data: students, courses, and
instructors. The instructor and course data is isolated to the courses file.
The student data is spread across all files. This document does not seek to be a
basic data dictionary. That is available in the file nsf-data-dict.xlsx.
Instead, this document seeks to elaborate on the characteristics of each of
these three entities using a data-driven approach.

# Students

The admissions data provides the snapshot of the student as an entering
freshman. Currently, no transfer students are present in the data. The data
contains information about each student at the time of admission. We also have a
record for each student for every term they have attended. This record includes
both cumulative information, as well as term-specific info. From the courses
file, we gain course-by-course granularity for each student. Finally, we have
graduation information for each student from the degrees file. There are a total
of *13,845 students* in the dataset.

## Overview

Collectively, these students span a total of 105 majors from 13 colleges. They
come from 2,341 known high schools and 57 known nations. Below are summary stats
on the numerical attributes we have for each student. For each attribute, only
the students with the attribute present are considered. The count statistic
gives the total number of students included in the calculation of the subsequent
stats. For the HSGPA, 21 students had a GPA greater than 5; these were excluded.

 Stat   HSGPA   SAT-1600   ACT Comp   TOEFL
 -----  ------  ---------  --------   -------
 count  13738   11633      3196       255
 mean   3.625    1138.756    25.817   566.216
 std    0.341     124.629    36.848    65.166
 min    2.330     630        10         7
 25%    3.395    1050        22       550
 50%    3.600    1130        24       577
 75%    3.850    1220        27       597
 max    4.977    1600      1370       667

## ACT Scores

 Stat    ACT Comp    ACT English   ACT Reading   ACT Math    ACT Science
 -----   ---------   -----------   -----------   ---------   -----------
 count   3196.000    3012.000      3012.000      3012.000    3011.000
 mean      25.817      24.584        25.424        24.328      23.949
 std       36.848       4.454         5.027         3.899       3.821
 min       10.000       5.000         7.000        14.000       5.000
 25%       22.000      22.000        22.000        22.000      21.000
 50%       24.000      24.000        25.000        24.000      24.000
 75%       27.000      27.000        30.000        27.000      26.000
 max     1370.000      36.000        36.000        36.000      36.000

## SAT Scores

 Stat    SAT-1600    SAT Verbal   SAT Math
 -----   ---------   ----------   ---------
 count   11633.000   11686.000    11689.000
 mean     1138.756     569.086      577.121
 std       124.629      76.828       72.992
 min       630.000     280.000      310.000
 25%      1050.000     520.000      520.000
 50%      1130.000     560.000      570.000
 75%      1220.000     620.000      630.000
 max      1600.000     800.000      800.000

## TOEFL Scores

 Stat    TOEFL Paper  TOEFL Int.   TOEFL All
 -----   ----------   ----------   ---------
 count    27.000      229.000      255.000
 mean    527.370       87.677      566.216
 std     155.436       15.312       65.166
 min       7.000       41.000        7.000
 25%     523.000       80.000      550.000
 50%     560.000       90.000      577.000
 75%     598.000       98.000      597.000
 max     667.000      118.000      667.000

## Military Status

 Category                 Count
 ---------------------    ------
 Civilian                 12,967
 Active Duty Dependent       507
 Surviving Dependent         331
 Veteran                      18
 Active Duty                  15
 Reserve/Nat.Guard             7

## Majors

4,601 students (33%) were undeclared at time of admission. The distribution of
those declared can be observed in Fig. 1 below. After admission, 3% of the
undeclared students declared a major, and some of those who were declared
shifted majors. The distribution at time of entry (first term) can be seen in
Fig. 2.

![Figure 1]()

![Figure 2]()

## Time to Graduation

These are the stats for the *2,967 students who graduated*. Missing values were
filled with the median, since very few values were missing.

 Stat    Years   CumGPA   # Terms   Avg Credits/Term   Summer Credits
 -----   -----   ------   -------   ----------------   --------------
 mean    4.044   3.325     9.212    13.352             4.027
 std     0.435   0.367     1.514     1.625             0.544
 min     1.5     2.160     3         6.000             1
 25%     4.0     3.070     8        12.200             4
 50%     4.0     3.340     9        13.444             4
 75%     4.0     3.600    10        14.707             4
 max     5.0     4.000    17        18.333             7

Of these students, *2,466 graduated in 4 years or less*. The stats for these
students are shown below. Missing values were again filled.

 Stat   Years     cumGPA    # Terms  Avg Credits/Term   Summer Credits
 ----   ------    -------   -------  ----------------   --------------
 mean    3.896    3.374      8.896   13.629             3.933
 std     0.290    0.348      1.310    1.482             0.482
 min     1.5      2.250      3        6.000             1
 25%     4.0      3.130      8       12.618             4
 50%     4.0      3.390      9       13.667             4
 75%     4.0      3.640     10       15.000             4
 max     4.0      4.000     17       18.333             7

There are few salient patterns that emerge when comparing the whole student
population to those that graduate in 4 years or less. Let us instead compare
against the *501 students* who took more than 4 years to graduate.

 Stat   Years    cumGPA    # Terms   Avg Credits/Term   Summer Credits
 ----   ------   -------   -------   ----------------   --------------
 mean   4.774    3.077     10.768    11.985             4.489
 std    0.249    0.372      1.491     1.606             0.595
 min    4.5      2.160      6         7.786             2
 25%    4.5      2.820     10        10.909             4
 50%    5.0      3.060     11        12.000             5
 75%    5.0      3.350     12        13.100             5
 max    5.0      3.980     15        16.875             6

Finally, the stats below are for students who graduated with a cumulative GPA of
3.5 or above (Deans List).

 Stat   Years    cumGPA    # Terms   Avg Credits/Term   Summer Credits
 ----   ------   -------   -------   ----------------   --------------
 mean   3.900    3.719      8.797     13.869            3.936
 std    0.416    0.142      1.447      1.438            0.617
 min    2.0      3.510      3          8.615            1
 25%    4.0      3.590      8         13.000            4
 50%    4.0      3.700      8         14.000            4
 75%    4.0      3.830      9         15.000            4
 max    5.0      4.000     15         17.833            7

