"""
IC 272 ASSIGNMENT-2
NIKHIL
B20219
8949463760
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import mean_squared_error

# reading the csv file
df = pd.read_csv("landslide_data3_miss (1).csv")

# 1
# making a dictionary with attributes as keys and count of missing values as the values
nan = df.isna().sum().to_dict()
# plotting bar graph between nan values count and respective attributes
plt.bar(nan.keys(), nan.values())
plt.title("Number of missing values for each attribute")
plt.xlabel("Attributes")
plt.ylabel("Number of missing values")
plt.xticks(fontsize=6)
plt.show()

# 2(a)
print("The number of rows to be dropped are", df['stationid'].isna().sum())
# counting the number of rows in column stationid which have missing values
df.dropna(subset=['stationid'], inplace=True)  # dropping the rows with missing values in column stationid

# 2(b)
# creating list of  number of missing values in each row
drop_row = df.isnull().sum(axis=1).tolist()
# we had to take 1/3 of attributes with missing values, stationid and dates had no missing values so we excluded them
attr = 7 / 3
print("The total number of tuples deleted are", sum(1 for i in drop_row if i > attr))
# keeping the rows which have at least 6 non NaN values
df = df.dropna(thresh=df.shape[1]-2, axis=0)

# 3
print("The number of missing values in each attributes are given below:")
# making a dictionary with attributes as keys and count of missing values as the values
miss_values = df.isnull().sum().to_dict()
print(miss_values)
print("The total number of missing values in the file", df.isnull().sum().sum())

# 4 (a)
miss_idx = {}  # index of rows with missing values
# we have to perform interpolation afterwards so we are not making changes in original dataframe
missm = df.copy()
col_names = missm.columns.values.tolist()  # making list of column names
col_names = col_names[2:]  # excluding columns dates and stationid as they dont have any missing values
for i in col_names:
    miss_idx[i] = missm.loc[pd.isna(missm[i]),
                  :].index.tolist()  # storing index of rows of respective columns with missing values
    missm[i] = missm[i].fillna(missm[i].mean())  # replacing missing values with mean

# 4 a. i)
missm_mode = []  # making list of modes of attributes
for i in col_names:
    missm_mode.append(round(float(missm[i].mode()), 3))
# creating a dataframe of mean, median, mode and standard deviation of attributes after replacing missing values with mean
missm_list = pd.DataFrame({'Attributes': col_names, 'Mean': [round(num, 3) for num in missm.mean().tolist()],
                           'Median': [round(num, 3) for num in missm.median().tolist()], 'Mode': missm_mode,
                           'Standard Deviation': [round(num, 3) for num in missm.std().tolist()]})
print("Mean,Median,Mode and Standard Deviation after being replaced by respective means:\n", missm_list)

org = pd.read_csv('landslide_data3_original (1).csv')  # reading original csv file
org_mode = []  # making list of modes of attributes of original file
for i in col_names:
    org_mode.append(round(float(org[i].mode()), 3))

# creating a dataframe of mean, median, mode and standard deviation of attributes in original file
org_list = pd.DataFrame({'Attributes': col_names, 'Mean': [round(num, 3) for num in org.mean().tolist()],
                         'Median': [round(num, 3) for num in org.median().tolist()], 'Mode': org_mode,
                         'Standard Deviation': [round(num, 3) for num in org.std().tolist()]})
print("Mean,Median,Mode and Standard Deviation of original file :\n", org_list)

Difference1=pd.DataFrame({'Attributes': col_names,'mean-':pd.Series(missm_list['Mean']-org_list['Mean']), 'median-':pd.Series(missm_list['Median']-org_list['Median']), 'mode-':pd.Series(missm_list['Mode']-org_list['Mode']), 'std-':pd.Series(missm_list['Standard Deviation']-org_list['Standard Deviation'])})
print('differences after replacing by respective means: \n', Difference1)
# 4 a. ii)
rmse_m = {}  # dictionary with attributes as keys and rmse as values
for i in col_names:
    actual = []  # values in orginal file
    predicted = []  # values that we replaced NaN with
    for j in miss_idx[i]:
        actual.append(org.loc[j, i])
        predicted.append(missm[i].mean())
    # calculating rmse of all attributes
    mse_m = mean_squared_error(actual, predicted)
    rmse_m[i] = round(math.sqrt(mse_m), 3)
print("RMSE by replacing with mean \n", rmse_m)
# plotting bar graph between rmse values and attributes
plt.bar(rmse_m.keys(), rmse_m.values())
plt.title("RMSE vs Attributes")
plt.xlabel("Attributes")
plt.ylabel("RMSE")
plt.yscale('log')
plt.xticks(fontsize=6)
plt.show()

# 4 (b)
df = df.interpolate(method='linear')  # replacing missing values with linear interpolation


# 4 b. i)
df_mode = []  # making list of modes of attributes
for i in col_names:
    df_mode.append(round(float(df[i].mode()), 3))

# creating a dataframe of mean, median, mode and standard deviation of attributes after replacing missing values with interpolation
df_list = pd.DataFrame({'Attributes': col_names, 'Mean': [round(num, 3) for num in df.mean().tolist()],
                        'Median': [round(num, 3) for num in df.median().tolist()], 'Mode': df_mode,
                        'Standard Deviation': [round(num, 3) for num in df.std().tolist()]})
print("Mean,Median,Mode and Standard Deviation after being replaced by interpolation:\n", df_list)
Difference2=pd.DataFrame({'Attributes': col_names, 'mean-':pd.Series(df_list['Mean']-org_list['Mean']), 'median-':pd.Series(df_list['Median']-org_list['Median']), 'mode-':pd.Series(df_list['Mode']-org_list['Mode']), 'std-':pd.Series(df_list['Standard Deviation']-org_list['Standard Deviation'])})
print('differences after replacing by interpolation: \n', Difference2)
# 4 b. ii)
rmse = {}  # dictionary with attributes as keys and rmse as values
for i in col_names:
    actual = []  # values in orginal file
    predicted = []  # values that we replaced NaN with
    for j in miss_idx[i]:
        actual.append(org.loc[j, i])
        predicted.append(df.loc[j, i])
    # calculating rmse of all attributes
    mse = mean_squared_error(actual, predicted)
    rmse[i] = round(math.sqrt(mse), 3)
# plotting bar graph between rmse values and attributes
print("RMSE by interpolation \n", rmse)
plt.bar(rmse.keys(), rmse.values())
plt.title("RMSE vs Attributes")
plt.xlabel("Attributes")
plt.ylabel("RMSE")
plt.yscale('log')
plt.xticks(fontsize=6)
plt.show()

# 5 a)
# Temperature
Q1_T = np.percentile(df['temperature'], 25, interpolation='midpoint')  # calculating first quartile
Q3_T = np.percentile(df['temperature'], 75, interpolation='midpoint')  # calculating third quartile
IQR_T = Q3_T - Q1_T  # calculating IQR
Upper_T = Q3_T + (1.5 * IQR_T)  # upper bound
Lower_T = Q1_T - (1.5 * IQR_T)  # lower bound
temp = df[(df['temperature'] < Lower_T) | (df['temperature'] > Upper_T)]  # storing the rows having outliers
print("The outliers in temperature are:", temp['temperature'].tolist())  # printing outliers of temperature

# plotting box plot before replacing the outliers
plt.boxplot(df['temperature'])
plt.title("Box Plot of temperature before replacing the outliers")
plt.xlabel("Temperature")
plt.ylabel("Values")
plt.show()

# Rain
Q1_R = np.percentile(df['rain'], 25, interpolation='midpoint')  # calculating first quartile
Q3_R = np.percentile(df['rain'], 75, interpolation='midpoint')  # calculating third quartile
IQR_R = Q3_R - Q1_R  # calculating IQR
Upper_R = Q3_R + (1.5 * IQR_R)  # upper bound
Lower_R = Q1_R - (1.5 * IQR_R)  # lower bound
Rain = df[(df['rain'] < Lower_R) | (df['rain'] > Upper_R)]  # storing the rows having outliers
print("The outliers in rain are:", Rain['rain'].tolist())  # printing outliers of rain

# plotting box plot before replacing the outliers
plt.boxplot(df['rain'])
plt.title("Box Plot of rain before replacing the outliers")
plt.xlabel("Rain")
plt.ylabel("Values")
plt.show()

# 5 b)
# Temperature
median_T = df['temperature'].median()  # median of the attribute
# replacing outliers with median
df.loc[df['temperature'] < Lower_T, 'temperature'] = median_T
df.loc[df['temperature'] > Upper_T, 'rain'] = median_T
# plotting box plot after replacing the outliers with median
plt.boxplot(df['temperature'])
plt.title("Box Plot of temperature after replacing the outliers")
plt.xlabel("Temperature")
plt.ylabel("Values")
plt.show()

# Rain
median_R = df['rain'].median()  # median of the attribute
# replacing outliers with median
df.loc[df['rain'] > Upper_R, 'rain'] = median_R
df.loc[df['rain'] < Lower_R, 'rain'] = median_R
# plotting box plot after replacing the outliers with median
plt.boxplot(df['rain'])
plt.title("Box Plot of rain after replacing the outliers")
plt.xlabel("Rain")
plt.ylabel("Values")
plt.show()
