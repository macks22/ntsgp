% DegreePlanner: Data Munging and Basic SVD For Grade Prediction
% Mack Sweeney
% March 26, 2015

# Overview

This document present preliminary results for the grade prediction task which
will inform the DegreePlanner (DP) application. The goal of the DP is to
recommend to students plans of study which maximize chances for success both in
the short term (next semester) and in the long term (graduate on time with a
good GPA and acquire desired skillset from program of study). The grade
prediction task will allow the DP to recommend courses which are suitable to a
student's skillset. Ultimately, we would like to filter the list of acceptable
courses using applicable domain knowledge. This might include information such
as:

*   Number of prerequisites satisfied (course options opened up)
*   Contribution to major requirements
*   Courses which student has the skills (prereqs.) for
*   Courses which interest the student
    -   For undecided students, we would like to guide their decision-making
        while keeping them moving forward in the majors they are most likely to
        select from.
    -   For decided students who are uncertain where their degree is taking
        them, we would like for them to be exposed to a variety of subfields.
    -   For decided students with clear direction, we want to ensure they don't
        miss the only offering in 2 years of that one course they are incredibly
        interested in taking.
*   ...

Once we have the list of suitable course options, we would ideally like to
recommend the courses in which the student will have the greatest success. While
the definition of success is left vague here, most students will be interested
in maximizing return on investment, which means graduating according to schedule
with grades that will move their career or future education goals forward.

This brings us to the current work: predicting student grades on courses which
they have not yet taken. Related work has achieved error as low as .15 (RMSE) on
this task (Elbadrawy et al., 2014). The model used to achieve this predictive
accuracy incorporated a variety of features, including information from the
college's learning management system (LMS). The results presented in this report
use ONLY historical grade information. The model learns entirely from a
collection of grade outcomes on courses, with no knowledge about the students or
the courses. The model also learns only from total course grades, while the
model built by Elbadrawy et al. also incorporates grades on individual
assignments and assessments within courses.

Using a simple Matrix Factorization (unbiased, regularized Singular Value
Decomposition), we are able to achieve an error of 0.953444. This means that we
can predict within plus or minus approximately 1 grade point. If a student is
going to get a B, we will predict somewhere between A and C. While this error is
unacceptable for the final DP system, the results are quite promising for this
basic prototype. This report describes the data processing (munging) tasks
completed to obtain several transformations of the grade data. The modeling
decisions made in the SVD model are also discussed. Finally, detailed results
for various transformations are presented, with discussion of the impact of
missing value policies, inclusion of bias terms, and the ratio of test to
training data used to learn the model.

# Data Munging

There are 20 unique grade types in the data. Below are counts for each:

 Grade  Count
 -----  -----
 A      78308
 B      50968
 A-     43414
 B+     36804
 C      25701
 B-     22960
 A+     18550
 C+     16249
 F      15810
 D      12747
 C-      5818
 W       5693
 S       4970
 NC      2319
 IN       461
 NR        90
 AU        51
 REG       43
 IP         9
 IX         1

## Missing Values

In order to use this data for a recommender system, we must determine how to
fill the missing values. From W down to IX, the values are recorded as 'nan'.
Below is the policy chosen for each:

 Grade    Policy
 -----    -----------------------------------
 W        mark as D (1.00 grade points)
 S        mark as B (mean of C to A range)
 NC       mark as D (middle of C- to F range)
 IN       discard
 NR       discard
 AU       discard
 REG      discard
 IP       discard
 IX       discard

Later we'll probably want to use mean value imputation for IN/NR/AU categories.
The meaning of the REG grade is currently unknown. A similar grade, ZREG, is
known to be applicable to study-abroad courses, so REG may be the same thing.
While these grades are still unknown, if they can be determined to indicate
study-abroad credit, we may be able to use them as indicators that a student has
taken courses abroad, which may be a useful binary feature for prediction.

## Unique IDs for Each Entity

In order to work with a data matrix of students and courses, we must assign
unique ids to all courses and map the student ids to contiguous 0-indexed ids.
The same is also done for instructors (for later processing). The TERMBNR
attribute is also mapped to an ordinal (0-14) scale, and the same is done for
the cohort. After filtering to only needed columns, we have:

    (sid, cid, termnum, cohort, iid, grdpts)

    sid:     student id
    cid:     course id
    termnum: term number (ordinal coding from `TERMBNR`)
    cohort:  term number (ordinal coding from `cohort`)
    iid:     instructor id
    grdpts:  quality points calculated from the grade

At this point, we can discard rows with missing grades and filter the rows so
that only the most recent grade for a particular course/student is kept. There
are a total of 407,517 records to start with. After filtering to most recent
grades, there are 377,864 (29,653 less). At this point, there are 59,376 mising
grades. After removing these records, we have 318,488 records to work with.

## Key Munging Decisions

The most influential munging decisions involved the following questions.

1.  How to deal with courses that have recitations/labs?
    +   Often have the same (DISC, CNUM) info, but specific SECTNO that
        identifies it as a recitation/lab
    +   Often no grade because cumulative grade is recorded in the lecture
    +   For now, we simply discard courses with no grades and treat any
        lab/recitation sections as separate courses, so long as they are
        recorded as accounting for a different number of credit hours (HRS).
2.  Do we only keep the most recent grade for a course?
    +   To start off, this is best for quick results.
    +   However, students who've taken a course more than once and done well
        the second time may be more familiar overall, and students who do
        worse the second time probably have serious prereq deficiencies or
        recurring study/persistence problems.
    +   This might be a good point to revisit later.
3.  How to do testing on course preprocessing?
    +   Should ideally have very few duplicate records after id mappings
    +   This should be verified to ensure the id mapping process does not merge
        records that should be separate or fail to assign the same entities to
        the same id.
    +   Some ad-hoc testing indicates this is not an issue; there are no test
        cases at this time, but some should be implemented before moving forward
        much further.
4.  How to split so that each course shows up at least once in training data?
    +   If we use cohorts 0-4 for training, there are 44 courses in the test
        set which never show up in the train set
    +   Some courses only show up once in the entire dataset.
    +   So we take this approach:
        1.  Split by specified cohorts/termnums.
        2.  Include first 3 instances of any course that ends up in test set but
            not in train set. Why 3? Arbitrary decision.

## Splitting Data for Train/Test Sets

When splitting the train/test set, the numbers of records per cohort will come
in handy. These are below.

 cohort   count
 ------   -------
     1    114661
     4    104677
     7     86504
     10    61068
     13    35685
     2      1426
     5      1157
     8       958
     11      654
     14      263

So here's how the splitting will work: the cohort value shown in the table below
is the last cohort in the range 0-\<cohort\> that makes up the training set. All
other cohorts make up the test set. There are two cohorts in an enrollment year,
and the summer terms have no cohorts. So for instance, when we say (later in
this document) that the train set is made up of the years 2009/2010, we mean
that we've taken all records for cohorts 0-4 and made those the test set and
taken all records from cohorts 5-14 and made that the test set. As seen below,
this is a 0.542:0.455 split. 54.2% of the data is used during training, and the
remaining 45.5% is used for testing.

Note that the 4th munging decision made (move courses unseen in training data
from test data) will shift these proportions slightly. The test set might end up
slightly smaller than expected and the training set slightly larger. In
practice, these deviations are small.

 TERMBNR   cohort  train   test
 -------  -------  -----   -----
 200940    0	   0.000   1.000
 200970    1	   0.282   0.718
 201010    2	   0.285   0.715
 201040    3	   0.285   0.715
 201070    4	   0.542   0.458
 201110    5	   0.545   0.455
 201140    6	   0.545   0.455
 201170    7	   0.758   0.242
 201210    8	   0.760   0.240
 201240    9	   0.760   0.240
 201270    10	   0.910   0.090
 201310    11	   0.912   0.088
 201340    12	   0.912   0.088
 201370    13	   0.999   0.001
 201410    14      1.000   0.000

## Working With libFM

Finally, we must determine how to set the parameters for libFM.

-   The -dim flag controls the use of bias terms and dimensionality D.
    +   ex: -dim '1,1,8' specifies: use global bias, use per-feature bias,
        and use 8 dimensions.
    +   We run experiments using both biased (global and per-feature), and
        unbiased SVD.
    +   We obtain results for D=5-20. Results below 5 were found to be generally
        inferior. Results throughout the range end up being quite similar in
        practice.
-   Specify stdev of initial Gaussian data spread (init_stdev).
    +   This influences the speed of learning.
    +   Values around 0.3 seem to work well.
-   Specify the number of iterations (iter).
    +   This can easily be adjusted after observing output to continue
        training for as long as necessary. Results tend to converge around 200
        iterations, so we use this setting for the experiments in this report.
    +   A time-constrained system (frequent updates, online learning) would need
        to discover a more theoretically optimal way of choosing the number of
        iterations (stop at a threshold, for instance).

# Results

The results below are for unbiased SVD, running MCMC for 200 iterations, using a
2009/2010 train set and 2011+ test set (0.542:0.458 split). We see the best
results with dim=17,7,15. The lowest error achieved is 1.01302.

When using bias terms (B-Train/B-Test), we see a slightly lower training error,
but a slightly higher test error. It seems likely that using bias terms with
MCMC increases the overfitting already demonstrated without bias terms. The best
dim values are 10,11,14, with the lowest error being 1.02133.

   D    Train       Test       B-Train     B-Test
 ---    --------    -------    --------    -------
  5	    0.684380	1.01764    0.667645	   1.02365
  6	    0.667926	1.01642    0.654794	   1.02252
  7	    0.653481	1.01435    0.642243	   1.02251
  8	    0.644737	1.02146    0.628243	   1.02238
  9	    0.632610	1.01995    0.619318	   1.02248
  10    0.622865	1.02076    0.609505	   1.02133
  11    0.610474	1.02188    0.595463	   1.02202
  12    0.604649	1.02006    0.585233	   1.02225
  13    0.592084	1.02512    0.574033	   1.02322
  14    0.576706	1.01735    0.562258	   1.02210
  15    0.571789	1.01641    0.555116	   1.02414
  16    0.557209	1.01767    0.543379	   1.02296
  17    0.545596	1.01302    0.533339	   1.02249
  18    0.537172	1.01758    0.522305	   1.02364
  19    0.532405	1.02333    0.514230    1.02262
  20    0.517350	1.01677    0.502459	   1.02263

We ran the same experiment with a 2009-2011 train to 2012+ test (0.758:0.242
split). We see the best results with dim=18,17,14. The results are slightly
worse, however, than when using the train/test split above, which is closer to
50/50 for train/test. The best error achieved is 1.03477. For biased SVD, the
best results are achieved with dim=15,14, with the lowest error being 1.03837.

   D    Train       Test       B-Train     B-Test
 ---    --------    -------    --------    -------
  5	    0.687828	1.04093    0.672179	   1.04021
  6	    0.675318	1.03845    0.658529	   1.04047
  7	    0.660240	1.03715    0.647662	   1.03927
  8	    0.651442	1.03882    0.634554	   1.04039
  9	    0.638006	1.04159    0.625403	   1.0396
  10	0.630670	1.04175    0.615111	   1.03949
  11	0.619764	1.03960    0.605737	   1.03999
  12	0.611302	1.03904    0.592758	   1.03919
  13	0.601583	1.04129    0.585550    1.04007
  14	0.590042	1.03691    0.573477	   1.03893
  15	0.582006	1.03698    0.566933	   1.03837
  16	0.571676	1.04058    0.559172	   1.04051
  17	0.562391	1.03563    0.547479	   1.04032
  18	0.548045	1.03477    0.541548	   1.04059
  19	0.539379	1.03694    0.529377	   1.04021
  20	0.533219	1.04054    0.518900    1.04075

The results achieved above for both splits included the W/S/NC students in the
training data as well as the test data. Intuitively, it may not be useful to
attempt to predict these grades, but it may still be useful to learn on them. If
we include them in the training data but not the test data, we get the following
results for the 2009/2010 train to 2011+ split.

The best unbiased results are with dim=12,20,19, with the lowest error being
0.960844. The best biased results are with dim=12,14,7, with the lowest error
being 0.966284. Once again, we see that the biased SVD has slightly higher
error, due to what appears to be increased overfitting of the training data.

   D    Train       Test       B-Train     B-Test
 ---    --------    --------   --------    --------
  5	    0.659399	0.967080   0.639938	   0.967926
  6	    0.643665	0.962952   0.629131	   0.966955
  7	    0.631757	0.968177   0.616369	   0.966649
  8	    0.622088	0.962710   0.603237	   0.967209
  9	    0.609059	0.961854   0.591818	   0.966694
  10	0.599347	0.965413   0.583238	   0.968292
  11	0.592024	0.965851   0.574805	   0.967108
  12	0.579132	0.960844   0.563687	   0.966284
  13	0.571948	0.966124   0.555227	   0.967807
  14	0.562994	0.969657   0.543442	   0.966396
  15	0.552263	0.966965   0.536085	   0.967036
  16	0.543606	0.968862   0.525741	   0.967031
  17	0.533092	0.965730   0.516282	   0.967503
  18	0.519940	0.962581   0.507756	   0.968649
  19	0.511244	0.961712   0.496931	   0.967299
  20	0.502875	0.961139   0.485402	   0.967849

We repeat the experiment with the 2009-2011 to 2012+ split. The unbiased version
gets the best results with dim=19,7,20, with lowest error of 0.981947. The
biased version gets the best results with dim=10,6,9, with lowest error of
0.985291.

   D    Train       Test       B-Train     B-Test
 ---    --------    -------    --------    --------
  5	    0.657692	0.984818   0.643650    0.986457
  6	    0.649395	0.987170   0.632349	   0.985428
  7	    0.635498	0.981992   0.620404	   0.986842
  8	    0.626715	0.986155   0.612259	   0.986049
  9	    0.615206	0.984155   0.602378	   0.985647
  10	0.608445	0.985714   0.591309	   0.985291
  11	0.597335	0.983227   0.582442	   0.985761
  12	0.590004	0.986533   0.573036	   0.986375
  13	0.582097	0.983119   0.563556	   0.986726
  14	0.572939	0.987374   0.554326	   0.987074
  15	0.563719	0.988552   0.544959	   0.986135
  16	0.553074	0.985233   0.534688	   0.986940
  17	0.543595	0.985413   0.528430    0.987052
  18	0.534158	0.983919   0.518219	   0.986682
  19	0.523937	0.981947   0.511827	   0.986972
  20	0.515118	0.982730   0.501050    0.986960

Finally, we should also determine whether the W/S/NC grades (as assigned by our
policy discussed above) actually lend any predictive power to the model. So the
results below leave these grades out of the training set as well as the test
set.

For the 2009/2010 train to 2011+ test: best unbiased results with dim=20,5,11,
with lowest error at 0.953444. Best biased results with dim=9,15,8, with lowest
error of 0.958892.

   D    Train       Test       B-Train     B-Test
 ---    --------    --------   --------    --------
  5	    0.634151	0.956057   0.616997    0.959190
  6	    0.622007	0.957995   0.603714    0.960581
  7	    0.610084	0.958705   0.591950    0.960839
  8	    0.597443	0.958651   0.579629    0.959253
  9	    0.587464	0.959980   0.570582    0.958892
  10	0.576084	0.959665   0.555875    0.960219
  11	0.561024	0.957015   0.546321    0.959632
  12	0.555260	0.960655   0.538043    0.960166
  13	0.546168	0.957860   0.529065    0.960024
  14	0.532881	0.959211   0.516884    0.961190
  15	0.521883	0.958170   0.508610    0.958924
  16	0.521518	0.962905   0.495598    0.960033
  17	0.504247	0.957437   0.490401    0.959950
  18	0.496581	0.964413   0.477844    0.960512
  19	0.489913	0.962628   0.467115    0.959606
  20	0.471555	0.953444   0.458063    0.960135

For the 2009-2011 train to 2012+ test: best unbiased results with dim=11,16,13,
with lowest error being 0.976809. Best biased results with dim=7,18,13, with
lowest error being 0.979664.

   D    Train       Test       B-Train     B-Test
 ---    --------    --------   --------    --------
  5     0.638519    0.982857   0.618547    0.980974
  6     0.625077    0.981294   0.607704    0.981628
  7     0.610387    0.980378   0.595524    0.979664
  8     0.604583    0.982069   0.585767    0.980608
  9     0.591842    0.979982   0.574585    0.979610
  10    0.583798    0.983484   0.566708    0.979998
  11    0.570425    0.976809   0.556857    0.980106
  12    0.562469    0.981154   0.545246    0.979965
  13    0.549713    0.978078   0.536388    0.979794
  14    0.542594    0.979121   0.527640    0.979868
  15    0.533473    0.980235   0.519600    0.980008
  16    0.523481    0.977375   0.506237    0.980053
  17    0.516066    0.979621   0.499707    0.980738
  18    0.505498    0.979736   0.492544    0.979646
  19    0.496339    0.978615   0.482982    0.979952
  20    0.491247    0.980589   0.473632    0.980732

# Conclusions

Clearly leaving out the uncertain grades (S/W/NC) is the best option among those
tried. The best result we get without those grades is from unbiased SVD on the
2009/2010 to 2011+ split (0.953444). The best result with the grades included in
the train set but not the test set also comes from unbiased SVD on the same
split (0.960844). We can conclude that either our policy for filling in grade
point values for these grades is poor, or that these grades don't have
significant predictive potential in the first place, since they are too noisy.
The first seems more likely, so it may be worth taking time to discover more
suitable policies.

In general, we notice that the inclusion of bias terms leads to slightly more
overfitting. This may be a result of insufficient data, or perhaps a result of
temporal trends or instructor trends that remain unaccounted for with the simple
SVD model. Future weeks of this project will include these features. So the
usefulness of bias terms will be explored after including each new feature in
order to determine when/if they begin to improve model accuracy.
