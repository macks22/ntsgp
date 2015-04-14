---
title: "DegreePlanner: The Prediction Task and Handling Cold Start"
author: Mack Sweeney
date: April 10, 2015
bibliography: week3.bib
csl: ieee.csl
geometry: margin=1.2in
---

# Introduction

The focus of previous weeks was on predicting grades for several subsequent
semesters using data from all prior semesters. This week we considered the
separate but related task of only predicting grades for the next semester given
data from all past semesters. In addition, we also explored the effects of
various cold-start handling strategies, both for student records and for course
records. Traditionally, these problems are known as user and item cold-start.
Accounting for these problems gives a clearer picture of the performance of
predictions. Both the next-term prediction task and the cold start problems
are discussed in more detail below.

## The Prediction Task

The DegreePlanner (DP) application will likely need both a short-term and
long-term view of how a student might perform across a variety of classes.
Hence, we can consider two separate but related prediction tasks. The first is
_next-term prediction_. The task is to predict the grades for courses a student
will take in the next term, given data from all previous terms. We only ever
predict one term ahead. This will be most useful for immediate ranking of course
offerings for the current term.

The second task is _all-term prediction_. In this case the goal is to predict
grades for all courses a student will take in the future (all subsequent terms),
given only the data from the currently known previous terms. In other words, if
the Fall 2010 term just finished, we have only 2009/2010 data to base our
predictions on. So we use the same dataset for predictions in 2011, 2012, 2013,
and 2014. This task is useful for long-term planning. It would be particularly
useful for building and updating plans of study. In general, these are
multi-year term-by-term listings of all courses which a student will need to
take for degree completion.

Both tasks are useful for a fully-featured DP application. Given the increased
data available for next-term prediction, we expect the error to be lower when
making these predictions. Additionally, we expect cold-start problems to be much
more detrimental to the all-term prediction problem.

## Cold Start Problems

Cold-start problems arise in collaborative filtering (CF) when the data
available for learning the model does not contain students or courses present in
the future for which we need to infer grades. For the DP application, both
problems will ideally be addressed. We want to provide guidance to newly
enrolled students, and we want to be able to make recommendations for newly
introduced courses. Using pure CF approaches, we cannot address cold-start in a
reasonable way. The models available simply do not handle it. Since there will
be a greater likelihood of new courses being introduced as time goes on, we
expect these issues become more prominent as we attempt to predict further and
further ahead. This means that pure CF models are unsuitable for the all-term
prediction task. However, they are still viable for next-term prediction. The
results below demonstrate this point.

In order to deal with cold-start, we will need to adopt more sophisticated
models. Content-based methods are capable of handling course cold-start.
Traditionally, the definition of content-based limits side information to the
items. So these models will still suffer from the student cold-start problem.
However, demographic models, which incorporate information about the users
(students), are capable of handling student cold-start. The differences between
these two classes of approaches have become blurred over time and now many
consider the category "content-based" to encompass both. libFm is capable of
emulating the following content-based methods:

*   SVD++ [@koren_factorization_2008]
*   TimeSVD++ [@koren_collaborative_2010]
*   kNN [@koren_factorization_2008],
    kNN++ [@koren_factorization_2008],
    and Factorized kNN [@koren_factor_2010]
*   Regression-based latent factor models
    [@agarwal_regression-based_2009, @gantner_learning_2010]

# Results and Analysis

Results are presented below. For the next-term prediction task, we include a
single table which lists the prediction RMSE for all terms, since the RMSE for
each term is the same regardless of the initial train/test split. For the
all-term prediction task, we include results for models trained on 2009-2010,
2009-2011, and 2009-2012 data. For these 4 tables, all records in the test were
filtered such that no grades were predicted for cold-start students or courses.
The number of predicted grades are shown for each term.

Since we are dealing with factorization machines (FMs), and not the actual
methods being emulated, we need to better understand the impact of cold-start
problems. We also include results similar to those above, with error computed
in the same way. The only difference is that the cold-start records have been
left in the test set. So we also make predictions for cold-start students and
courses.

For all results, the dimensionality was set to 7, since that number yielded good
results in prior experiments on this dataset. 200 iterations were used after
initializing the data with Gaussian noise, using a standard deviation of 0.5.
Results are presented for each of the 6 methods we can currently emulate.

\break

## The Prediction Task

### Next-Term Prediction

 method              all      term1      term2      term3      term4      term5      term6      term7
 -------------    -------   -------    -------    -------    -------    -------    -------    -------
 # test records    247300       170      10946        577      10385      20773       1388      19854
 BiasedBPTF       0.79797   0.90396    0.86529    0.81610    0.79026    0.80068    0.84221    0.76903
 BiasedTimeSVD    0.79824   0.90396    0.86381    0.81610    0.79026    0.80068    0.83968    0.77018
 BiasedSVD        0.79854   0.90396    0.86381    0.80438    0.79026    0.80068    0.83968    0.77018
 BPTF             0.80595   2.95873    0.87840    0.83281    0.80755    0.80215    0.86280    0.78255
 SVD              0.80603   2.95873    0.87840    0.83281    0.80755    0.80215    0.86280    0.78255
 TimeSVD          0.80603   2.95873    0.87840    0.83281    0.80755    0.80215    0.86280    0.78255


 method             term8      term9     term10     term11     term12     term13     term14
 -------------    -------    -------    -------    -------    -------    -------    -------
 # test records     30598       2261      29481      38637       3130      33703      45397
 BiasedBPTF       0.79091    0.83793    0.77741    0.76276    0.82069    0.78959    0.84367
 BiasedTimeSVD    0.79091    0.83715    0.77731    0.76351    0.81898    0.79091    0.84365
 BiasedSVD        0.79341    0.83715    0.77750    0.76351    0.81921    0.79091    0.84365
 BPTF             0.79729    0.85351    0.78473    0.76786    0.83551    0.79177    0.84809
 SVD              0.79773    0.85020    0.78440    0.76851    0.83593    0.79177    0.84804
 TimeSVD          0.79773    0.85020    0.78440    0.76851    0.83551    0.79436    0.84613

### All-term Prediction, 2009-2010 Train Set

 method           all         term5      term6      term7      term8      term9
 -------------    -------   -------    -------    -------    -------    -------
 # predictions        486        39          5         57         50          5
 BPTF             0.87352   0.78520    0.76290    0.80796    0.86192    0.75851
 SVD              0.87702   0.79044    0.71279    0.81861    0.86541    0.77919
 BiasedTimeSVD    0.87886   0.83390    0.71224    0.81222    0.87968    0.74139
 TimeSVD          0.88093   0.80062    0.74724    0.83255    0.87871    0.76409
 BiasedBPTF       0.88802   0.82319    0.67461    0.81961    0.89467    0.80468
 BiasedSVD        0.89187   0.81370    0.74742    0.82464    0.87801    0.77307

\break

 method            term10     term11     term12     term13     term14
 -------------    -------    -------    -------    -------    -------
 # predictions        101         86          6         67         70
 BPTF             0.84292    0.96696    0.80321    0.97979    0.83417
 SVD              0.85521    0.97145    0.73879    0.97943    0.82911
 BiasedTimeSVD    0.83957    0.98592    0.71662    0.97283    0.82844
 TimeSVD          0.84334    0.97012    0.78062    0.98039    0.84261
 BiasedBPTF       0.85754    0.98437    0.70793    0.98713    0.84243
 BiasedSVD        0.85855    1.00830    0.69108    0.99231    0.84498

### All-term Prediction, 2009-2011 Train Set

 method           all         term8      term9     term10     term11     term12     term13     term14
 -------------    -------   -------    -------    -------    -------    -------    -------    -------
 # predictions        472        46          6        113        111         10         95         91
 SVD              0.89729   0.86323    0.72179    0.74383    0.90270    0.67243    1.15199    0.86884
 BPTF             0.89980   0.85545    0.74308    0.74774    0.91308    0.69297    1.15478    0.86172
 TimeSVD          0.89980   0.85545    0.74308    0.74774    0.91308    0.69297    1.15478    0.86172
 BiasedTimeSVD    0.90849   0.88555    0.74266    0.76556    0.91263    0.66404    1.15666    0.87121
 BiasedSVD        0.91020   0.89132    0.69958    0.74653    0.91873    0.66694    1.18185    0.86960
 BiasedBPTF       0.91199   0.86983    0.76917    0.76429    0.92567    0.65880    1.17659    0.86101

### All-term Prediction, 2009-2012 Train Set

 method           all        term11     term12     term13     term14
 -------------    -------   -------    -------    -------    -------
 # predictions        297        62         14        110        111
 BiasedSVD        0.87695   0.80151    0.94223    0.91682    0.87134
 BiasedBPTF       0.87995   0.80692    0.94121    0.91475    0.87853
 BiasedTimeSVD    0.87995   0.80692    0.94121    0.91475    0.87853
 BPTF             0.88450   0.80108    0.89675    0.92231    0.89209
 SVD              0.88450   0.80108    0.89675    0.92231    0.89209
 TimeSVD          0.88450   0.80108    0.89675    0.92231    0.89209

\break

### Comparison

To summarize, the best results in each of the 4 tables are:

 method        RMSE      # Predictions
 -----------   -------   -------------
 BiasedBPTF    0.79797          247300
 BPTF          0.87352             472
 SVD           0.89729             486
 BiasedSVD:    0.87695             297

As we can clearly see, the all-term prediction task is essentially worthless if
we remove cold-starts. The vast majority of records in subsequent terms end up
having unseen students and courses. Meanwhile, the next-term prediction RMSE is
reasonable for the baseline methods being used. It is useful to further examine
the results for this task.

 method           500-RMSE    200-RMSE     150-RMSE     100-RMSE      50-RMSE
 -------------   ---------   ---------    ---------    ---------    ---------
 BiasedBPTF        0.79684     0.79797      0.79924      0.79924      0.80134
 BiasedTimeSVD     0.79696     0.79824      0.79930      0.79930      0.80110
 BiasedSVD         0.79700     0.79854      0.79895      0.79895      0.80132
 BPTF              0.80044     0.80595      0.82011      0.82011      0.86638
 SVD               0.80020     0.80603      0.82155      0.82155      0.86928
 TimeSVD           0.80031     0.80603      0.82129      0.82129      0.86552

If we look only at the predictions made after 500 iterations of model updates,
all methods are equally accurate if we round to the first decimal, and the
biased methods outperform the unbiased methods by only 0.01 if we round to 2
decimal places. However, we can still make some useful observations.

All methods that use bias terms outperform those that don't, and all the methods
that incorporate time equal or outperform those that don't to 3 decimal places.
We can hypothesize that the bias terms are capturing useful information. Since
per-student, per-course, per-term (for methods that use time), and a global bias
term are being used, it is not possible to say specifically what information is
being captured. Further analysis on this point might prove fruitful. However, we
can say that methods that use bias terms converge to useful predictions much
faster than those that don't. Even after 50 iterations, the biased methods are
nearing the accuracy achieved at 500 iterations. At 100 iterations, the error
differs only after the 2nd decimal. In contrast, the methods that don't use bias
terms take about 200 iterations to converge to predictions with accuracy similar
to that obtained after 500 iterations.

The differences between the methods that use time and those that don't are
small. So it seems the term number by itself is not a particularly interesting
feature, although it does seem to be slightly informative. We might consider
changing the term number feature currently being used to the number of months
elapsed since the start of the dataset. We might also consider incorporating
other temporal features related to content once content-based techniques are
employed.

## Cold Start Problems

### Next-Term Prediction

 method           all          term1      term2      term3      term4      term5      term6      term7
 -------------    -------    -------    -------    -------    -------    -------    -------    -------
 # predictions     309091      11815      11143        628      21702      20938       1454      31709
 BiasedTimeSVD    0.83189    0.97572    0.87471    0.80732    0.85974    0.80214    0.84285    0.83367
 BiasedSVD        0.83249    0.97572    0.87471    0.82286    0.85974    0.80300    0.83017    0.83367
 BiasedBPTF       0.83254    0.97572    0.87471    0.82286    0.85974    0.80214    0.83017    0.83367
 SVD              0.91679    2.95161    0.89183    0.84559    0.87625    0.80851    0.87012    0.85389
 TimeSVD          0.91743    2.95161    0.89183    0.84559    0.87625    0.80851    0.87012    0.85389
 BPTF             0.91772    2.95161    0.89183    0.84559    0.87864    0.80881    0.87012    0.85389
 
 method             term8      term9     term10     term11     term12     term13     term14
 -------------    -------    -------    -------    -------    -------    -------    -------
 # predictions      30774       2325      41603      38830       3197      47350      45623
 BiasedTimeSVD    0.79683    0.83582    0.82756    0.76469    0.81796    0.85359    0.84636
 BiasedSVD        0.79741    0.83582    0.82932    0.76469    0.81796    0.85453    0.84724
 BiasedBPTF       0.79741    0.83582    0.82932    0.76560    0.81796    0.85453    0.84724
 SVD              0.80357    0.84693    0.83762    0.77387    0.83878    0.86005    0.84924
 TimeSVD          0.80357    0.84693    0.83874    0.77387    0.83878    0.86260    0.84985
 BPTF             0.80357    0.84693    0.83762    0.77387    0.83959    0.86424    0.84985

### All-term Prediction, 2009-2010 Train Set

 method           all         term5      term6      term7      term8      term9
 -------------    -------   -------    -------    -------    -------    -------
 # predictions     141524       178         85      12211      11573        613
 TimeSVD          0.95851   0.78951    0.67307    0.92352    0.96598    1.02999
 BiasedBPTF       0.95923   0.80828    0.68864    0.92567    0.96800    1.04630
 BPTF             0.95938   0.80179    0.67706    0.92475    0.96608    1.03200
 SVD              0.95994   0.79002    0.67224    0.92397    0.96564    1.02699
 BiasedSVD        0.96021   0.80266    0.68339    0.92625    0.96930    1.04799
 BiasedTimeSVD    0.96085   0.80080    0.68340    0.92744    0.96994    1.04493

\break

 method            term10     term11     term12     term13     term14
 -------------    -------    -------    -------    -------    -------
 # predictions      23207      21966       1502      35149      35040
 TimeSVD          0.93203    0.94323    0.94552    0.95920    0.99551
 BiasedBPTF       0.93288    0.94497    0.96206    0.95891    0.99453
 BPTF             0.93349    0.94389    0.95354    0.96010    0.99583
 SVD              0.93272    0.94415    0.95374    0.96106    0.99803
 BiasedSVD        0.93333    0.94587    0.95894    0.96024    0.99579
 BiasedTimeSVD    0.93507    0.94658    0.95919    0.96068    0.99574

### All-term Prediction, 2009-2011 Train Set

 method           all         term8      term9     term10     term11     term12     term13     term14
 -------------    -------   -------    -------    -------    -------    -------    -------    -------
 # predictions      75226       200         87      12516      11786        640      25115      24882
 TimeSVD          0.97681   1.24816    0.74636    0.93870    0.96476    0.96852    0.97240    1.00496
 BPTF             0.97848   1.25286    0.74285    0.94053    0.96733    0.98268    0.97394    1.00594
 BiasedTimeSVD    0.97858   1.25323    0.76817    0.94062    0.96876    0.98639    0.97361    1.00567
 BiasedSVD        0.97884   1.25186    0.76158    0.94060    0.97060    0.99367    0.97357    1.00549
 BiasedBPTF       0.97894   1.24630    0.75806    0.93983    0.96987    0.99573    0.97378    1.00631
 SVD              0.98195   1.24209    0.75509    0.94363    0.97031    0.98461    0.97686    1.01050

### All-term Prediction, 2009-2012 Train Set

 method           all        term11     term12     term13     term14
 -------------    -------   -------    -------    -------    -------
 # predictions      28301       224         98      14062      13917
 BiasedBPTF       1.00853   0.92101    0.84930    0.99187    1.02789
 BiasedSVD        1.00853   0.92101    0.84930    0.99187    1.02789
 BiasedTimeSVD    1.00853   0.92101    0.84930    0.99187    1.02789
 BPTF             1.00927   0.92825    0.83819    0.99378    1.02742
 SVD              1.00927   0.92825    0.83819    0.99378    1.02742
 TimeSVD          1.00927   0.92825    0.83819    0.99378    1.02742

\break

### Comparison

To summarize, the best results in each of the 4 tables are:

 method          RMSE      # Predictions
 -------------   -------   -------------
 BiasedTimeSVD   0.83189          309091
 TimeSVD         0.95851          141524
 TimeSVD         0.97681           75226
 BiasedBPTF      1.00853           28301

Here the differences in number of predictions are uninteresting, since they are
due entirely to the original train/test split for the all-term prediction task.
Instead, it is interesting to note that the jump in RMSE between the next-term
and all-term prediction task is slightly greater than when cold-start records
are present. In general, this is a harder task. However, it is surprising that
the results are as good as they are without the use of bias terms. We do see
bias terms start to have a positive impact on the last training set, and this is
likely due to the larger training corpus to learn meaningful biases from.

For next-term prediction, the bias terms are much more beneficial than they are
in the all-term prediction. It is likely these end up providing reasonable
estimates for cold-start records. We do not see the same effect on the all-term
prediction task, in which biased methods only outperform unbiased methods
consistently on the last train/test set. This is likely because bias terms can
only be learned for courses and students who have been seen before (not
cold-start). Only in the last train/test split will the biases be learned for
most of the courses and students, so it is only in those results that we see
their benefit. Further, we expect bias terms to be useful only if they can be
averaged over a decent number of training examples. Otherwise we expect their
usefulness to degrade significantly as we predict for terms further in the
future.

Perhaps the most interesting observation to make here is that these error
results aren't much higher. As seen in the first 4 tables, the best RMSE for
next-term prediction is 0.79797, and the best for all-term prediction is
0.87352. The results obtained here are only slightly better. So either these
methods are lousy in general, or they are surprisingly good at dealing with
cold-start issues. This observation motivates the need for a random baseline. We
can make predictions from a Gaussian distribution with mean of 2 and standard
deviation of 1. We can round values above 4 to 4 and values below 0 to 0. The
RMSE of these predictions will serve as the simplest basline to compare all
other methods to.

# Moving Forward

We will incorporate our newly motivated random baseline into our results. We
will start incorporating course content and student demographics into the
models. In particular, we want to discover how much predictive information can
be gleaned from:

1.  Instructor info:
    *   instructor id
    *   average grade in all courses taught up to the current term
    *   department
2.  Course info
    *   prerequisites: grades the student has gotten in them
    *   prerequisites: whether a student has taken them or not (binary)
    *   discipline
    *   level (1=100,2=200,3=300,4=400)
    *   average grade of students up to current term
3.  Student demographics
    *   race
    *   age
    *   gender
    *   zip code
    *   major

# References

\noindent
\vspace{-2em}
\setlength{\parindent}{-0.25in}
\setlength{\leftskip}{0.25in}
\setlength{\parskip}{15pt}
