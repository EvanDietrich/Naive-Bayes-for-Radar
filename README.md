**********************************
Author:   Evan Dietrich
Course:   Comp 131 - Intro AI
Prof:     Santini

Assign:   Naive Bayesian Classification
Date:     12/01/2020
File:     README.md
**********************************

Overview:
This program implements a Naive Bayesian Classifier to categorize flying objects
captured from radar track data into bird (B) and aircraft (A) categories. This
program also allows for the inclusion of additional features for the bonus.

Technical Breakdown:
radar.py   - Runs the program.  
README.md  - Instructions to run program and assignment overview.

Running the Program:
To run the program, ensure all downloaded files are on your machine, and run
"python3 radar.py" on your terminal screen. This will start the program
for a automated running procedure. The procedure given is controlled by the
global variables (essentially, parameter options for runtime) found in the 
'IMPORTS + GLOBALS' section of the 'radar.py' file. The example run is with:

FILE = 'data.txt'
N_FOLDS = 10
ADDTL_FEATURES = False

By editing the globals, you can test the Bayes Classifier on different fold
numbers and cross-validate model performance. This allows the user to test our
model's ability in predicting new data not used in estimation, allowing us to
identify selection bias and/or overfitting, as well as let us know how well the
model works on an unknown dataset.

When you run the program, you will noticed 10 different simulation-performances,
providing insight to model performance, as well as an overall accuracy percent.
When running with N_FOLDS = 10, avg accuracy tended towards roughly 70%
performance, without the use of additional feature engineering.

BONUS SECTION ANALYSIS::
In completing the bonus part of this assignment, I extracted additional values
beyond the general speed (allowing for mean and variance calculations). To see
this action, run the program with the altered global, which takes effect in
the preprocessingData function:

ADDTL_FEATURES = True

This quickly extracted feature allows testing the variability of each 
categorized datapoint by comparing it to the datapoint prior. I had a hunch that
the bird category was more likely to have an un-smooth trajectory and have
many more quick jumps and dips in overall velocity, compared to planes, which
are much heavier and cannot make such drastic shifts in regular, clean data. I
also considered the use of more continuous feature engineering processes, such
as calculating the acceleration from the given dataset over time, to do the same
thing, but I found that this method could generally start to account for the
same idea. Although, it would be interesting to test out different acceleration
values over various continual time spans, instead of these discrete and
back-to-back datapoints to calculate the likelihood of a "drastic shift" value.

I implemented another version of this by saying that shifts over a certain
value would be considered a boolean "1" while those under the threshold would
be given "0". This did not alter performance too much so I left the initial
approach as is.

The inclusion of this additional "drastic shift" feature increased model
performance over the same # of cross-fold validations. Running this multiple
times, avg accuracy maintained an approximate 84% over trials, beating out
the roughly 70% avg accuracy performance without the use of extra features.


Collaborators:
I completed this assignment with assistance from the student posts and
instructor answers on the Comp131 Piazza page, with the lecture material,
our class textbook, and an online article on the use of different distribution
formats: "https://inverseprobability.com/talks/notes/naive-bayes.html", which
provided some insight into other ways to test model performance, such as with
confusion matrices, as well as methods for additional smoothing.


Notes:
Testing my solution by altering the number of folds did not seem to greatly
affect avg prediction accuracy, while makes sense that the model is not 
varying greatly in its output and speaks for overall mathematical continuity.
This was maintained with the additional feature inclusion as well.
