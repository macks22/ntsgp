% DegreePlanner: Incorporating Temporal Information in Grade Prediction
% Mack Sweeney
% April 3, 2015

# Motivation

When predicting grades, we expect time to be a critical component of the
prediction. The same course can change drastically over the course of several
years. This may correspond to a curriculum overhaul, departmental changes in
content focus, or simply a new set of professors teaching the course. We also
expect the students to change to some extent. This may be due to more stringent
admissions requirements for particular cohorts, or it could simply be due to
external factors we are not aware of in high school curriculums.

In a more subtle sense, we might expect students to be more capable of
succeeding if they have access to knowledgeable peers in their cohort. For
instance, cohort 2009 may happen to have numerous students who find employment
as peer advisors and are extremely proficient in teaching stastics and
mathematics concepts. In such a case, we might find that students from that
cohort or the cohort immediately following achieve higher grades in statistics.
This specific example is unlikely to have a significant impact on the overall
cohort performance, but several factors like this could have a net effect which
is significant.

We might also consider less obvious temporal information, such as how long a
student has been enrolled, how many courses the student has taken, and how many
terms they have taken classes in. Although we are not considering student
attributes yet, once these are incorporated we will also want to evaluate the
predictive power of features such as cumulative GPA. We can incorporate time
into this evaluation by also looking at a time-weighted GPA which places more
weight on recent grades. This would capture the hypothesis that a student's most
recent semester's grades will likely be somewhat more indicative of their
near-term performance.

These observations lead us to expect time will be very informative for
predicting grades. This report will start to incorporate temporal information by
incorporating knowledge about the term in which each grade was obtained into the
model. The experience and results obtained from this analysis will facilitate
incorporation of additional temporal information in future work on the
DegreePlanner application.

# Encoding Term Numbers

Each of the 15 enrollment term codes is encoded ordinally as $x_i = i,
i=0...14$. In this way, we ensure the encoding is ordered temporally, such that
every term has a smaller value than any later term and a larger value than any
earlier term.

 TERMBNR  number
 -------  -------
 200940    0
 200970    1
 201010    2
 201040    3
 201070    4
 201110    5
 201140    6
 201170    7
 201210    8
 201240    9
 201270    10
 201310    11
 201340    12
 201370    13
 201410    14

# Methods for Incorporation

libFM provides two recommended ways for incorporating temporal information,
which roughly correspond to two prior models which are approximated by the
resulting factorization machines.

1.  Categorical encoding: TimeSVD
    *   With this approach, we include a single new feature in each
        student/course feature vector. This feature encodes the term and has the
        same feature number across all student vectors. For student s and course
        c, the value of the term feature is the term in which student s (last)
        took course c.
    *   This approach corresponds to the TimeSVD model developed by Koren
        (2010) [1].
2.  One-hot encoding: Bayesian Probabilistic Tensor Factorization (BPTF)
    *   With this approach, we one-hot encode the terms, such that each
        student/course feature vector gets 15 new binary features, one for each
        term. So if student s took course c in term i, then the feature value
        for term i will be set to 1 and the feature values for the other 14
        terms will be set to 0.
    *   This approach corresponds to the time-aware BPTF model developed by
        Xiong et al. (2010) [2].

# Results

Results are shown below for both the TimeSVD (categorical) and the BPTF
(one-hot) encoding methdos on the 2009/2010 train to 2011+ test set. Results on
the 2009-2011/2012+ split were inferior, as they were for the simple SVD model
we are building upon. Both biased and unbiased results are shown; biased methods
are prefaced with a 'b'. All results are ordered from best to worst. For
reference, the best results obtained using plain SVD on this train/test split
had an error of 0.958892.

 Method      D    Train       Test
 --------  ---    --------    --------
 BPTF	    17	  0.509178	  0.952962
 bBPTF	    15	  0.503588	  0.954709
 bBPTF	    18	  0.479188	  0.954783
 bBPTF	    20	  0.466602	  0.955553
 BPTF	    9	  0.583910	  0.957339
 BPTF	    8	  0.591355	  0.957395
 bTimeSVD	8	  0.593131	  0.975528
 bTimeSVD	6	  0.620513	  0.987056
 TimeSVD	7	  0.609017	  0.988637
 bTimeSVD	7	  0.626521	  1.079010
 TimeSVD	8	  0.642374	  1.082760
 TimeSVD	6	  0.665636	  1.123230

# Discussion


We can clearly see that BPTF outperforms TimeSVD on this data, and the inclusion
of bias terms start to appear useful with this model. The best results are with
plain BPTF and 17 dimensions. We achieve a reduction in error of 0.005930. While
nice to see, this is quite a bit lower than we would expect to be able to obtain
by incorporating time. It seems necessary to examine the prediction error in
more detail to determine if perhaps the inclusion of time is more benefical for
certain future terms and less useful for others.

# References

[1] Y. Koren, “Collaborative Filtering with Temporal Dynamics,” Commun. ACM,
    vol. 53, no. 4, pp. 89–97, Apr. 2010.  
[2] L. Xiong, X. Chen, T.-K. Huang, J. G. Schneider, and J. G. Carbonell,
    “Temporal Collaborative Filtering with Bayesian Probabilistic Tensor
    Factorization.,” in SDM, 2010, vol. 10, pp. 211–222.
