'''
Nikhil
------
B20219
------
8949463760
'''

import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# reading the csv files of training and testing dataset
train = pd.read_csv("SteelPlateFaults-train.csv")
test = pd.read_csv("SteelPlateFaults-test.csv")
y_true = test['Class'].values
# dropping the attributes
train = train.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
test = test.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
# separating the training data based on class
train0 = train.groupby('Class').get_group(0).to_numpy()
train0 = np.delete(train0, 23, axis=1)
train1 = train.groupby('Class').get_group(1).to_numpy()
train1 = np.delete(train1, 23, axis=1)
# dropping the class attribute
train = train.drop(['Class'], axis=1)
test = test.drop(['Class'], axis=1)
# number of n_components
Q = [2, 4, 8, 16]
for q in Q:
    y_pred = []
    # building gmm
    gm0 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5).fit(train0)
    gm1 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5).fit(train1)
    # computing the weighted log probabilities
    log0 = gm0.score_samples(test) + np.log(len(train0)/len(train))
    log1 = gm1.score_samples(test) + np.log(len(train1)/len(train))
    for i in range(len(log0)):
        if log0[i] > log1[i]:
            y_pred.append(0)

        else:
            y_pred.append(1)
    print("The confusion matrix for Q =", q, "is\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", round(100 * accuracy_score(y_true, y_pred), 3),"\n")

# question 2 of part A is in report
