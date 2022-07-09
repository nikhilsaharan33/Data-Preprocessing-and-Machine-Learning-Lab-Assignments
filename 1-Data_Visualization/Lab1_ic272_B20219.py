# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 18:24:08 2021

@author: nikhi
"""

"""
Name: Nikhil 
Roll No.:B20219
Mobile No.: 8949463760
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
t1=time.time()
df = pd.read_csv('pima-indians-diabetes.csv')

#1

# Mean, Median, Maximum, Minimum, Standard deviation of all attributes except 'class'
means=df.mean(axis=0)
medians=df.median(axis=0)
max=df.max(axis=0)
min=df.min(axis=0)
std=df.std(axis=0)
properties=pd.DataFrame({'Mean':means,"Median":medians,'Maximum':max,'Minimum':min,"Standard Deviation":std})
properties=properties.drop('class')  # dropping class attribute
print(properties.round(3))

# Mode of all attributes except 'class'
print("\n Mode of all attributese except 'class'\n ")
modes = (df.mode(axis=0, dropna=True))
modes = modes.drop(axis=0, columns='class')  # dropping class attribute😁😁😁
print(modes.round(3))

#2(a)

print('Scatter plot: Age (in years) vs. pregs')
plt.scatter(df['Age'], df['pregs'])
plt.xlabel('Age (in years)')
plt.ylabel('Number of times pregnant')
plt.title('Scatter plot: Age (in years) vs. pregs')
plt.show()

print('Scatter plot: Age (in years) vs. plas')
plt.scatter(df['Age'], df['plas'])
plt.xlabel('Age (in years)')
plt.ylabel('Plasma glucose concentration')
plt.title('Scatter plot: Age (in years) vs. plas')
plt.show()

print('Scatter plot: Age (in years) vs. pres')
plt.scatter(df['Age'], df['pres'])
plt.xlabel('Age (in years)')
plt.ylabel('Diastolic blood pressure (mm Hg)')
plt.title('Scatter plot: Age (in years) vs. pres(mm Hg)')
plt.show()

print('Scatter plot: Age (in years) vs. skin')
plt.scatter(df['Age'], df['skin'])
plt.xlabel('Age (in years)')
plt.ylabel('Triceps skin fold thickness (mm)')
plt.title('Scatter plot: Age (in years) vs. skin (mm)')
plt.show()

print('Scatter plot: Age (in years) vs. test')
plt.scatter(df['Age'], df['test'])
plt.xlabel('Age (in years)')
plt.ylabel('test(in mm U/mL)')
plt.title('Scatter plot: Age (in years) vs. test(in mm U/mL)')
plt.show()

print('Scatter plot: Age (in years) vs BMI')
plt.scatter(df['Age'], df['BMI'])
plt.xlabel('Age (in years)')
plt.ylabel('Body mass index (kg/(m)^2)')
plt.title('Scatter plot: Age (in years) vs. BMI (kg/(m)^2)')
plt.show()

print('Scatter plot: Age (in years) vs pedi')
plt.scatter(df['Age'], df['pedi'])
plt.xlabel('Age (in years)')
plt.ylabel('Diabetes pedigree function')
plt.title('Scatter plot: Age (in years) vs. pedi')
plt.show()

#2(b)

print('Scatter plot: BMI (in kg/m2) vs. pregs')
plt.scatter(df['BMI'], df['pregs'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('Number of times pregnant')
plt.title('Scatter plot: BMI (in kg/m2) vs. pregs')
plt.show()

print('Scatter plot: Age (in years) vs. plas')
plt.scatter(df['BMI'], df['plas'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('Plasma glucose concentration ')
plt.title('Scatter plot: BMI (in kg/m2) vs. plas')
plt.show()

print('Scatter plot: BMI (in kg/m2) vs. pres')
plt.scatter(df['BMI'], df['pres'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('Diastolic blood pressure (mm Hg)')
plt.title('Scatter plot: BMI (in kg/m2) vs. pres(mm Hg)')
plt.show()

print('Scatter plot: BMI (in kg/m2) vs. skin')
plt.scatter(df['BMI'], df['skin'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('Triceps skin fold thickness (mm)')
plt.title('Scatter plot: BMI (in kg/m2) vs. skin (mm)')
plt.show()

print('Scatter plot: BMI (in kg/m2) vs. test')
plt.scatter(df['BMI'], df['test'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('2-Hour serum insulin (mu U/mL)')
plt.title('Scatter plot: BMI (in kg/m2) vs. test (mu U/mL)')
plt.show()

print('Scatter plot: BMI (in kg/m2) vs pedi')
plt.scatter(df['BMI'], df['pedi'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('Diabetes pedigree function')
plt.title('Scatter plot: BMI (in kg/m2) vs. pedi')
plt.show()

print('Scatter plot: BMI (in kg/m2) vs BMI')
plt.scatter(df['BMI'], df['Age'])
plt.xlabel('BMI (in kg/m2)')
plt.ylabel('Body mass index (weight in kg/(height in m)^2)')
plt.title('Scatter plot: BMI (in kg/m2) vs. Age (in years)')
plt.show()

#3(a)

# Correlation coefficient of Age with pregs
print("The value of correlation coefficient of Age with pregs :", df['Age'].corr(df['pregs']).round(3))

# Correlation coefficient of Age with plas
print("The value of correlation coefficient of Age with plas :", df['Age'].corr(df['plas']).round(3))

# Correlation coefficient of Age with pres
print("The value of correlation coefficient of Age with pres :", df['Age'].corr(df['pres']).round(3))

# Correlation coefficient of Age with skin
print("The value of correlation coefficient of Age with skin :", df['Age'].corr(df['skin']).round(3))

# Correlation coefficient of Age with test
print("The value of correlation coefficient of Age with test :", df['Age'].corr(df['test']).round(3))

# Correlation coefficient of Age with BMI
print("The value of correlation coefficient of Age with BMI :", df['Age'].corr(df['BMI']).round(3))

# Correlation coefficient of Age with pedi
print("The value of correlation coefficient of Age with pedi :", df['Age'].corr(df['pedi']).round(3))

#3(b)

# Correlation coefficient of BMI with pregs
print("The value of correlation coefficient of BMI with pregs :", df['BMI'].corr(df['pregs']).round(3))

# Correlation coefficient of BMI with plas
print("The value of correlation coefficient of BMI with plas :", df['BMI'].corr(df['plas']).round(3))

# Correlation coefficient of BMI with pres
print("The value of correlation coefficient of BMI with pres :", df['BMI'].corr(df['pres']).round(3))

# Correlation coefficient of BMI with skin
print("The value of correlation coefficient of BMI with skin :", df['BMI'].corr(df['skin']).round(3))

# Correlation coefficient of BMI with test
print("The value of correlation coefficient of BMI with test :", df['BMI'].corr(df['test']).round(3))

# Correlation coefficient of BMI with Age
print("The value of correlation coefficient of BMI with Age :", df['BMI'].corr(df['Age']).round(3))

# Correlation coefficient of BMI with pedi
print("The value of correlation coefficient of BMI with pedi :", df['BMI'].corr(df['pedi']).round(3))

#4
print('histogram of attribute pregs')
bins1=np.arange(0,30,2)
plt.hist(df['pregs'],bins=bins1, rwidth=2)
plt.show()
print('histogram of attribute skin')
plt.hist(df['skin'])
plt.show()

#5
print('histogram of attribute ‘pregs’ for each of the 2 classes individually')
df1=df.groupby('class')
df2=df1.get_group(0)
df3=df1.get_group(1)
plt.hist(df2['pregs'],bins=bins1, rwidth=2)
plt.show()
plt.hist(df3['pregs'],bins=bins1, rwidth=2)
plt.show()
#6
plt.boxplot(df['pregs'])
plt.show()
plt.boxplot(df['plas'])
plt.show()
plt.boxplot(df['pres'])
plt.show()
plt.boxplot(df['skin'])
plt.show()
plt.boxplot(df['test'])
plt.show()
plt.boxplot(df['BMI'])
plt.show()
plt.boxplot(df['pedi'])
plt.show()
plt.boxplot(df['Age'])
plt.show()
t2=time.time()
print(t2-t1)