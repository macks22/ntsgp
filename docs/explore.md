% DegreePlanner Exploratory Phase Overview
% Mack Sweeney

# Purpose of this Document

This document presents an overview of the exploratory phase for the
DegreePlanner (DP) system. This phase consists of three main thrusts: (1) the
visualization of the available data, (2) the identification of descriptive
patterns in the data, and (3) the development of initial prototypes for grade
prediction in future courses.

# Visualization

There are three primary entities in our data. We have students, instructors, and
courses. There are currently six different visualizations planned which will
help explore and understand these three entities. With this understanding, more
informed decisions can be made when building initial system prototypes. The
first three visualizations outlined will explore courses, the next two will
explore students, and the last will explore the relation between students and
instructors.

[InCHlib](http://openscreen.cz/software/inchlib/use_cases/12) will be used for
the interactive heat maps and
[sigma.js](http://hansifer.com/sigmaArrowDemo1.htm) will be used for the network
visualizations. The heat maps support clustering visualization using
dendrograms and onclick functions which can show futher detail regarding a
particular block in the map. sigma.js supports visualization of both directed
and undirected graphs and scaling of nodes and edges based on attributes.

## Interactive Course Heatmap

*   Courses will be shown on the y-axis
*   Clusters will be at the discipline (department), major, and minor levels
*   Course attributes will be shown on the x-axis
    -   difficulty (as measured by grade distribution)
    -   course level (100-400, with higher being more intense color)
    -   bottleneck extent (how much of a bottleneck?)
    -   honeypot extent (how much of a honeypot?)
    -   number of students (average, normalized)
    -   number of sections (per term, on average, normalized)
    -   instructor variety (number of instructors per year)
    -   credit hours
    -   typical instructor experience (rank/tenure/class info)
    -   extent to which students are in same discipline as course
*   The onclick attribute for each course will be the course grade distribution

## Major Flow Graph

*   Nodes are majors
*   Edges represent transitions between major of first node to major of second
*   Size of node proportional to number of students in major
*   Size of edges proportional to number of transitions
*   Include a node for incoming students, drop outs, and graduating

## Prerequisite Tree

*   Nodes are courses
*   Edges from one course to another if the 1st is a prerequisite for the 2nd
*   Node attributes will be discipline, course #, title, description
*   May also want to include parallel edges between corequisites
*   Questions
    -   how to denote known substitutes?
    -   how to link courses to all possible prereqs for very broad option (e.g.
        any CS 300 level)
*   Bottlenecks can be characterized qualitatively by out-degree

## Interactive Student Heatmap

*   Place students on y-axis
*   Cluster by college, major, minor
*   On x-axis:
    -   cumGPA
    -   SAT/ACT
    -   TOEFL
    -   high school GPA
    -   academic standing (determine ordering, worse is more intense color)
    -   number of terms attended
    -   avg # courses per term
*   onclick function can be student GPA trendline
*   Other options
    -  may want to do another heatmap with GPA over time (time on x-axis)
    -  may also consider clustering by cohort

## Co-enrollment Network

*   Nodes are students
*   Edges between students if they have taken a course together
*   Edge weight is number of times co-enrolled
*   Draw initial edges based on shared high school
    -   perhaps weight this higher and color differently

## Bipartite Student Instructor Network

*   Two node types: students and instructors
*   Edge connecting students to professors if student has been taught by
    instructor
*   Edge weight representing number of occurences
*   Node/edge sizes:
    -   Number of courses taught by instructor (total)
    -   Number of courses taken by student (total)
*   Visualize using a [Radial-Circular
    layout](http://melihsozdinler.blogspot.com/2010/03/graph-of-day-17-political-networks.html)
    with [hierarchical edge
    bundling](http://mbostock.github.io/d3/talk/20111116/bundle.html)
    -   d3 has these capabilities

# Descriptive Pattern Identification

In order to better understand the data from a quantitative standpoint, it will
be useful to accumulate a set of descriptive statistics that describe the data
at a high level. We will also engage in some initial feature engineering,
including identification of course and instructor "bottlenecks" and "honeypots"
and accumulation of course prerequisite information. We will also use standard
measures of correlation to cast some light on possible value of predictors for
subsequent feature selection/engineering in initial prototoypes.

## Basic Descriptive Stats

Per term & year statistics:

*   Students
    -   GPA distribution
    -   distribution over colleges and majors
    -   counts of students in each academic standing
    -   distribution of credit hours earned
    -   Number graduated, per major and per college
*   Courses
    -   how many
    -   how many per college and discipline
    -   grade distribution
*   Instructors
    -   how many
    -   avg. number of courses taught
    -   group by class/rank/tenure and count

Overall statistics:

*   Students
    -   how many total at a given time
    -   how many coming in at each term
    -   how many per cohort
    -   distribution over colleges and majors
    -   GPA trends, both overall and major GPA
    -   distribution over high schools
    -   distribution over region of origin (zip/nation)
    -   distribution of SAT/ACT/TOEFFL (ignore missing)
    -   % who are score optional
*   Courses
    -   total courses offered
    -   distribution of course offerings over colleges and disciplines
    -   overall grade distribution
    -   avg. grade trendline per course, then split across disciplines
    -   number of prereqs (may change across years, so do by year)
*   Instructors
    -   how many
    -   avg. # courses taught per term
    -   number of students taught total
    -   avg. number of students taught per term
    -   grade distribution for courses taught
    -   std for avg. grade per instructor
    -   distribution over class/rank/tenure

## Bottlenecks and Honeypots

Both courses and instructors may be considered to be "bottlenecks" or
"honeypots." Specifically, a course may be empirically determined to be a
bottleneck in two ways. Either students take the course, fail, and change majors
or drop out, or a students fail a course one or more times before passing and
continuing in the same major. Qualitatively, we can also define course
bottlenecks in a major as those that serve as a prerequisite for many other
required courses in that major.

The idea of a honeypot is related to the first empirical definition of a
bottleneck. If a student takes a course outside his/her discipline, does very
well, and then switches his/her major to the discipline of that course, it is a
honeypot.

The same notion of bottleneck and honeypot can be extended to instructors by
considering the success/failure of students to be related to instructor
pedagogy. Hence a student switching majors to/from the discipline of the course
taught by the instructor determines how much of a honeypot/bottleneck that
instructor is.

These two notions may serve as useful data points by themselves. However, when
combined, they will likely yield even more interesting patterns. For instance, a
student taking a bottleneck and a honeypot at the same time may be even more
likely to leave his/her current major to join the major of the honeypot.

## Correlations

Basic correlations across the dataset will also be useful to characterize the
data and tag certain attributes as seeming to be more/less important for
predictive capabilities of others. For instance, we may want to see how the high
school GPA, SAT, ACT, and TOEFL correlate with first and/or second term GPA
after enrolling.

# Initial Prototypes

The initial prototypes developed will attempt to predict student grades for
future courses. Three different models have been conceptualized. The first is a
matrix factorization based approach that builds on the work of Koren [1]. The
third model involves a multi-linear regression approach which builds from the
work of Karypis et al. [2]. The third model will utilize sequential pattern
mining to identify trajectories of (course, grade) pairs that are predictive of
future (course, grade) pairs.

## Matrix Factorization Model

The idea here is to use the SVD++ algorithm as our baseline. The implementation
in libFM will be used. We can do some feature engineering at this stage to get a
sense of predictive value of features and get a feel for how well matrix
factorization based methods will perform. From here, we can add more interesting
variables, such as bias terms and temporal aspects.

## Multi-Linear Regression Model

In contrast to matrix factorization, MLR offers increased personalization via
the use of individual student membership in each regression model. This provides
a way to quantitatively segment students based on varying predictors of success.
Such an approach may be useful to discover interesting variations in success
predictors. This method can also be improved by adding an instructor bias term
and incorporating additional temporal dynamics, such as in Koren [3].

## Sequential Pattern Mining Model

The idea for this model is to mine sequences of (course, grade) pairs that lead
up to particular (course, grade) outcomes for a specified course. Then we can
say that the ideal path to success/failure in a particular course is comprised
of a particular sequence of (course, grade) pairs preceding it. We may also
discover that a variety of such ideal paths exist. This model would provide a
particularly useful pattern for the goal of recommending courses for students to
take.

# References

[1](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
    Y. Koren, R. Bell, and C. Volinsky. Matrix factorization techniques for recommender
    systems. IEEE Computer, 42(8):30–37, 2009.

[2](http://glaros.dtc.umn.edu/gkhome/node/1124)
    A. Elbadrawy, R. Studham, and G. Karypis. Personalized Multi-Regression
    Models for Predicting Students' Performance in Course Activities. UMN CS
    14-011, 2014.

[3](http://dl.acm.org/citation.cfm?id=1557072)
    Y. Koren, “Collaborative Filtering with Temporal Dynamics,” Proc. 15th ACM
    SIGKDD Int’l Conf. Knowledge Discovery and Data Mining (KDD 09), ACM Press,
    2009, pp. 447-455.
