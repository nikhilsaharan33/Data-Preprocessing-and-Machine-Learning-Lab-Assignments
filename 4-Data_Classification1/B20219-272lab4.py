# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 23:43:22 2021

@author: nikhil
"""

""" Nikhil B20219 8949463760"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1 Splitting the dataset into training and testing sets
# reading csv file
data = pd.read_csv("SteelPlateFaults-2class.csv")
# storing column names
column = data.columns.to_numpy()
# splitting the dataset according to class
group = data.groupby('Class')
# converting the class data into numpy array
class_0 = group.get_group(0).to_numpy()
class_1 = group.get_group(1).to_numpy()
# splitting the classes dataset into training and testing sets
train_0, test_0 = train_test_split(class_0, test_size=0.30, random_state=42, shuffle=True)
# Since we got to have same number of training samples for both classes so we take 70% samples from the smaller class which turns out to be 57.6% in the bigger class(class 1)...
train_1, test_1 = train_test_split(class_1, test_size=0.30, random_state=42, shuffle=True)
# combining the class wise splitted dataset
train = np.concatenate((train_0, train_1), axis=0)
test = np.concatenate((test_0, test_1), axis=0)
# converting the splitted data into dataframe
training = pd.DataFrame(train, columns=column)
testing = pd.DataFrame(test, columns=column)
# saving the testing and training data as csv files
training.to_csv('SteelPlateFaults-train.csv', index=False)
testing.to_csv('SteelPlateFaults-test.csv', index=False)
# splitting training dataset into its attributes and labels
X_train = training.iloc[:, :-1].values
y_train = training.iloc[:, training.shape[1] - 1].values
# splitting testing dataset into its attributes and labels
X_test = testing.iloc[:, :-1].values
y_test = testing.iloc[:, testing.shape[1] - 1].values

print("Q1:")
print("\n")
# given values of k
K = [1, 3, 5]
highestaccuracy = 0
mostaccurate_k = 1
for k in K:
    # classifying the test tuples based on values of k
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    # predicting the class
    y_pred = classifier.predict(X_test)
    # (a) finding the confusion matrix for each k
    print("The confusion matrix for K=", k, "is\n", confusion_matrix(y_test, y_pred))
    print("\n")
    # noting the value of K for which the accuracy is high
    if accuracy_score(y_test, y_pred) > highestaccuracy:
        mostaccurate_k = k
        highestaccuracy = accuracy_score(y_test, y_pred)
    # finding classification accuracy for each k
    print("The accuracy score for K=", k, "is", round(accuracy_score(y_test, y_pred), 3))
    print("\n")

print("K=", mostaccurate_k, "has highest accuracy of", round(highestaccuracy, 3))
print("\n")

# 2
# Normalising training dataset in range [0-1]
norm_train = training.copy()
norm_test = testing.copy()
max_train = {}
min_train = {}

for i in column:
    if i == 'Class':
        continue
    else:
        min_train[i] = min(norm_train[i])
        max_train[i] = max(norm_train[i])
        new_mna = 0
        new_mxa = 1
        old_train = norm_train[i].values.tolist()
        new_train = []
        old_test = norm_test[i].values.tolist()
        new_test = []
        for value in old_train:
            x = ((value - min_train[i]) * (new_mxa - new_mna) / (max_train[i] - min_train[i])) + new_mna
            new_train.append(x)

        for value in old_test:
            x = ((value - min_train[i]) * (new_mxa - new_mna) / (max_train[i] - min_train[i])) + new_mna
            new_test.append(x)

        norm_train[i] = norm_train[i].replace(old_train, new_train)
        norm_test[i] = norm_test[i].replace(old_test, new_test)

norm_train.to_csv('SteelPlateFaults-train-Normalised.csv', index=False)
norm_test.to_csv('SteelPlateFaults-test-normalised.csv', index=False)

print("Q2:")
print("\n")
# splitting training dataset into its attributes and labels
X_norm_train = norm_train.iloc[:, :-1].values
y_norm_train = norm_train.iloc[:, norm_train.shape[1] - 1].values
# splitting testing dataset into its attributes and labels
X_norm_test = norm_test.iloc[:, :-1].values
y_norm_test = norm_test.iloc[:, norm_test.shape[1] - 1].values
# given values of k
K = [1, 3, 5]
accuracy_norm = 0
accurate_norm_k = 1
for k in K:
    # classifying the test tuples based on values of k
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_norm_train, y_norm_train)
    # predicting the class
    y_norm_pred = classifier.predict(X_norm_test)
    # (a) finding the confusion matrix for each k
    print("The confusion matrix for K=", k, "is\n", confusion_matrix(y_norm_test, y_norm_pred))
    print("\n")
    # noting the value of K for which the accuracy is high
    if accuracy_score(y_norm_test, y_norm_pred) > accuracy_norm:
        accurate_norm_k = k
        accuracy_norm = accuracy_score(y_norm_test, y_norm_pred)
    # finding classification accuracy for each k
    print("The accuracy score for K=", k, "is", round(accuracy_score(y_norm_test, y_norm_pred), 3))
    print("\n")

print("K=", accurate_norm_k, "has highest accuracy of", round(accuracy_norm, 3))
print("\n")

# 3
# Building a Bayes Classifier
print("Q3", "\n")
training = training.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
testing = testing.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
# splitting training dataset into its attributes and labels
X_train = training.iloc[:, :-1].values
y_train = training.iloc[:, training.shape[1] - 1].values
# splitting testing dataset into its attributes and labels
X_test = testing.iloc[:, :-1].values
y_test = testing.iloc[:, testing.shape[1] - 1].values

# sample mean and covariance for class 0
training_0 = pd.DataFrame(train_0, columns=column)
training_0 = training_0.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
X_train_0 = training_0.iloc[:, :-1].values
y_train_0 = training_0.iloc[:, training_0.shape[1] - 1].values
mean_0 = np.mean(X_train_0, axis=0)
cov_0 = np.cov(X_train_0.T)
column_num = [x for x in range(1, 24)]
matrix_0 = pd.DataFrame(X_train_0, columns=column_num)
covariance_0 = pd.DataFrame(matrix_0.cov().T.round(decimals=3))
covariance_0.to_csv('covariance_0.csv')
print("Mean of class 0:\n", [round(x, 3) for x in mean_0])
print("\n")
# sample mean and covariance for class 1
training_1 = pd.DataFrame(train_1, columns=column)
training_1 = training_1.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Y_Minimum', 'X_Minimum'], axis=1)
X_train_1 = training_1.iloc[:, :-1].values
y_train_1 = training_1.iloc[:, training_1.shape[1] - 1].values
mean_1 = np.mean(X_train_1, axis=0)
cov_1 = np.cov(X_train_1.T)
matrix_1 = pd.DataFrame(X_train_1, columns=column_num)
covariance_1 = pd.DataFrame(matrix_1.cov().T.round(decimals=3))
covariance_1.to_csv('covariance_1.csv')

print("Mean of class 1:\n", [round(x, 3) for x in mean_1])
print("\n")
# calculating prior probabilities for both classes
prior0 = len(X_train_0) / len(X_train)
prior1 = len(X_train_1) / len(X_train)


# defining the likelihood function
def probability_x(x, mean, cov):
    power = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return (np.exp(power)) / ((2 * np.pi) ** 11 * (np.linalg.det(cov)) ** 0.5)


# calculate the likelihood and predicting class according to it
y_bayes_pred = []
for x in X_test:
    liklihood0 = probability_x(x, mean_0, cov_0) * prior0
    liklihood1 = probability_x(x, mean_1, cov_1) * prior1
    if liklihood0 > liklihood1:
        y_bayes_pred.append(0)
    else:
        y_bayes_pred.append(1)

print("The confusion matrix for Bayes model is\n", confusion_matrix(y_test, y_bayes_pred))
print("\n")
print("The accuracy for Bayes model is", round(accuracy_score(y_test, y_bayes_pred), 3))
print("\n")

# 4
print("Q4:")
# Tabulating the best results of all three classifiers
comparison = {"S. No.": [1, 2, 3], "Classifier": ["KNN", "KNN on normalised data", "Bayes"],
              "Accuracy (in %)": [round(100 * highestaccuracy, 3), round(100 * accuracy_norm, 3),
                                  round(100 * accuracy_score(y_test, y_bayes_pred), 3)]}

table = pd.DataFrame(comparison)
print("Comparison between classifiers based upon classification accuracy:\n", table)
print("\n")
