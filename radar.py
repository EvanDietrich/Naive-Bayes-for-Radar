################################################################################
# Author:   Evan Dietrich
# Course:   Comp 131 - Intro AI
# Prof:     Santini
#
# Assign:   Naive Bayesian Classification
# Date:     12/01/2020
# File:     radar.py
################################################################################

################################################################################
#       IMPORTS + GLOBALS
################################################################################
FILE = 'data.txt'
N_FOLDS = 10
ADDTL_FEATURES = False

import os
import sys
import math
import random
import numpy as np
import pandas as pd

from math import pi
from math import exp
from math import sqrt
from csv import reader
from random import seed
from random import randrange

################################################################################
#       MODEL FUNCTIONALITY + HELPERS
################################################################################

# Mean & St. Dev math functions while ensuring type(float)
def mean(nums):
    return sum(nums)/float(len(nums))
 
def stdev(nums):
    return sqrt(sum([(i-mean(nums))**2 for i in nums])/float(len(nums)-1))

# Modifies newly imported data, removing NaN values, returning new structure
def preprocessData(data):
    new_col, end_lst = ['B','A','B','B','B','A','A','A','A','B'], []
    data.insert(loc=300, column='Category', value=new_col)
    a_indx, b_indx, a_lst, b_lst = [1, 5, 6, 7, 8], [0, 2, 3, 4, 9], [], []
    data = data.values.tolist()

    for i, sublist in enumerate(data):
        for val in sublist:
            if i in a_indx:
                check_float = isinstance(val, float)
                if check_float == True:
                    a_lst.append(val)
            else:
                check_float = isinstance(val, float)
                if check_float == True:
                    b_lst.append(val)

    a_lst, b_lst = [i for i in a_lst if i != 0], [i for i in b_lst if i != 0]

    prior_vals = 0
    for i, vals in enumerate(a_lst):
        if ADDTL_FEATURES == True:
            temp = vals
            hold = abs(prior_vals - temp)
            end_lst.append([vals, hold, 'A'])
            prior_vals = vals
        else:
            end_lst.append([vals,'A'])

    prior_vals = 0
    for i, vals in enumerate(b_lst):
        if ADDTL_FEATURES == True:
            temp = vals
            hold = abs(prior_vals - temp)
            end_lst.append([vals, hold, 'B'])
            prior_vals = vals
        else:
            end_lst.append([vals,'B'])
    return end_lst

# String Data -> Int Data
def convertToInt(data, col):
    category = [row[col] for row in data]
    indiv, temp = set(category), dict()
    for i, value in enumerate(indiv):
        temp[value] = i
    for row in data:
        row[col] = temp[row[col]]
    return temp
 
# Appraises Bayes overall prediction ability. Combines measures of fitness
# in prediction as user-instructs w/number of folds.
def appraiseBayes(data, run, N_FOLDS, *args):
    scores = []
    split, copy_data, fold_size = [], list(data), int(len(data)/N_FOLDS)

    for _ in range(N_FOLDS):
        fold = []
        while len(fold) < fold_size:
            idx = randrange(len(copy_data))
            fold.append(copy_data.pop(idx))
        split.append(fold)

    for fold in split:
        training, testing = list(split), []
        training.remove(fold)
        training = sum(training, [])
        for row in fold:
            temp = list(row)
            testing.append(temp)
            temp[-1] = None
        predicted = run(training, testing, *args)
        actual = [row[-1] for row in fold]
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        scores.append(correct/float(len(actual)) * 100.0)
    return scores

# Concentrates relevant data
def condense(data):
    results = [(mean(col), stdev(col), len(col)) for col in zip(*data)]
    del(results[-1])
    return results
 
# Splits training data by category, determines statistical peformance
def statsCategory(data):
    split, results = dict(), dict()
    for i in range(len(data)):
        vector = data[i]
        category = vector[-1]
        if (category not in split):
            split[category] = list()
        split[category].append(vector)

    for category, rows in split.items():
        results[category] = condense(rows)
    return results
 
# Calculate Probability Distribution Function for input
def calcPDF(input, mean, stdev):
    return (1/(sqrt(2 * pi) * stdev)) * exp(-((input-mean)**2/(2 * stdev**2)))
 
# Probability of predicting correct category
def probCategory(results, row):
    num_rows = sum([results[label][0][2] for label in results])
    probabilities = dict()
    for category, class_summaries in results.items():
        probabilities[category] = results[category][0][2]/float(num_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[category] *= calcPDF(row[i], mean, stdev)
    return probabilities
 
# Actual predicted category per row
def predictCategory(results, row):
    probabilities = probCategory(results, row)
    guess, likelihood = None, -1
    for category, probability in probabilities.items():
        if guess is None or probability > likelihood:
            likelihood, guess = probability, category
    return guess
 
# Runs the Naive Bayes Model + returns category prediction
def naiveBayesModel(train, test):
    results, predictions = statsCategory(train), []
    for row in test:
        output = predictCategory(results, row)
        predictions.append(output)
    return predictions
 
################################################################################
#       MAIN PROGRAM
################################################################################

def runRadar():
    # Import from FILE provided in Global Header, cleans data for our purposes
    data = pd.read_csv(FILE, sep=",", header=None).fillna(0)
    data = preprocessData(data)
    convertToInt(data, len(data[0])-1)

    # Test Bayes Model Performance
    scores = appraiseBayes(data, naiveBayesModel, N_FOLDS)
    print("**********************************************")
    print('Simulation scoring: %s' % scores)
    print('Average Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    print("**********************************************")

# Runtime Header: Begins Radar Routine (to classify flying objects from data)
if __name__ == '__main__':
    print("\n>>> Running ---Naive Bayes Classifier--- on: '" + FILE +"'" + "\n")
    if ADDTL_FEATURES == True:
         print(">>> Additional Features included\n")
    runRadar()
