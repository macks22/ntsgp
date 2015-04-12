% DegreePlanner: The Prediction Task and Handling Cold Start
% Mack Sweeney
% April 3, 2015

# Introduction

The focus of previous weeks was on predicting grades for several subsequent
semesters using data from all prior semesters. This week we considered the
separate but related task of only predicting grades for the next semester given
data from all past semesters. In addition, we also briefly explored the effects
of both course and student cold start problems. Accounting for these problems
gives a clearer picture of the performance of predictions. Both the term-by-term
prediction task and the cold start problems are discussed in more detail below.

## The Prediction Task

## Cold Start Problems

# Analysis

## The Prediction Task

Compare results from overall prediction to term-by-term. You don't currently
have results for each term based on the overall prediction, which is what is
really needed here. Mention this and lay out the term-by-term results anyway to
illustrate the difference. For now, limit comparisons to overall scoring.
Hypothesize regarding why the term-by-term scores are better. Also think about
and discuss why BPTF produces some of the best results for the long-term
predictions but some of the worst for the term-by-term results.

The term-by-term predictions across the 4 train/test splits is another good
point to discuss. Calculate a running average instead of just the global average
you have now. Use this for a comparison across data splits.

## Cold Start Problems

Run the long-term and term-by-term prediction tasks on a variety of data sets.
Include all combinations of:

student: [0, 1, 2, 3, 4, 5]
course:  [0, 1, 2, 3, 4, 5]

This will be 30 different runs. Include overall test RMSE for each method on
each setting. This will be 6 x 2 (both prediction tasks) x 30 results. The
combos should be on the first column, the method on the next, and the task on the
last. This will be 4 columns, with the results on the last. These should all be
sorted in ascending order of RMSE.

It will also be really useful to know how many students and courses are actually
being back-filled at each setting. Perhaps this can be output to a separate
file and just stored in a temporary directory? This should go into the table in
a column after the (student, course) settings in the first column.

# Discussion

# Moving Forward

Discuss ideas such as parameter sweep (number of dimensions), and other ideas
for temporal data, such as including the months instead of just a term number.
Also the idea of including past course indicators, either binary or grade
points.

# References

[1] Y. Koren, “Collaborative Filtering with Temporal Dynamics,” Commun. ACM,
    vol. 53, no. 4, pp. 89–97, Apr. 2010.  
[2] L. Xiong, X. Chen, T.-K. Huang, J. G. Schneider, and J. G. Carbonell,
    “Temporal Collaborative Filtering with Bayesian Probabilistic Tensor
    Factorization.,” in SDM, 2010, vol. 10, pp. 211–222.
