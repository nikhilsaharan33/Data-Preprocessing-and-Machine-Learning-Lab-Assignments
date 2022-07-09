# -*- coding: utf-8 -*-

'''NIKHIL B20219 8949463760'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# 1-(a)
# reading the csv file
cases = pd.read_csv('daily_covid_cases.csv')
# generating the x-ticks
x = [16]
for i in range(10):
    x.append(x[i] + 60)

# listing labels for the Month-Year
labels = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']
original = cases['new_cases']
plt.figure(figsize=(20, 10))
plt.xticks(x, labels)
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.title('Cases vs Months')
plt.plot(original)
plt.show()

# 1-(b)
print('1(b):')
# generating time series with 1 day lag
lagged = cases['new_cases'].shift(1)
corr = pearsonr(lagged[1:], original[1:])
print("The autocorrelation coefficient between the generated one-day lag time sequence and the given time sequence =",
      round(corr[0], 3))


# 1-(c)
# scatter plot between one day lagged sequence and given time sequence
plt.xlabel('given time sequence')
plt.ylabel('one day lagged time sequence')
plt.title('one day lagged sequence vs. given time sequence')
plt.scatter(original[1:], lagged[1:])
plt.show()

# 1-(d)
print('1(d):')
# lag values
lag = [1, 2, 3, 4, 5, 6]
correlation = []
print("The correlation coefficient between each of the generated time sequences and the given time sequence:")
for d in lag:
    lagged = cases['new_cases'].shift(d)
    corr = pearsonr(lagged[d:], original[d:])
    correlation.append(corr[0])
    print(f"{d}-day =", round(corr[0],3))
# line plot of correlation coefficients vs lagged values
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.title('Correlation coefficients vs Lagged Values')
plt.plot(lag, correlation)
plt.show()

# 1-(e)
sm.graphics.tsa.plot_acf(original,lags=lag)
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.show()

#2
# Train test split
series = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# 2-(a)
print('2(a):')
# training the model
p = 5
model = AutoReg(train, lags=p)
# fit/train the model
model_fit = model.fit()
# Get the coefficients of AR model
 
coef = model_fit.params
coef = np.array(coef) 
coef = np.around(coef,3)
# printing the coefficients
print('The coefficients obtained from the AR model are', coef)

# 2-(b)
#using these coefficients walk forward over time steps in test, one step each time
history = train[(len(train)-p):]
history = [history[i] for i in range(len(history))]
predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
for t in range(len(test)):
  length = len(history)
  Lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0] # Initialize to w0
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] # Add other values 
  obs = test[t]
  predicted.append(yhat) #Append predictions to compute RMSE later
  history.append(obs) # Append actual test value to history, to be used in next step.


# 2(b)-(i)
# scatter plot between actual and predicted values
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs Predicted values')
plt.scatter(predicted, test)
plt.show()

# 2(b)-(ii)
# line plot between actual and predicted values
plt.figure(figsize=(20, 10))
plt.xlabel('Days')
plt.ylabel('New Cases') 
plt.title('Predicted and Actual values')
plt.plot(test)
plt.plot(predicted)
plt.show()

# 2(b)-(iii)
print('2(b):')
# computing rmse
rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
print('RMSE(%):',rmse_per)

# computing MAPE
mape = np.mean(np.abs((test - predicted)/test))*100
print('MAPE:',mape)

# 3
print('3:')
lag_val = [1,5,10,15,25]
RMSE = []
MAPE = []
for l in lag_val:
  model = AutoReg(train, lags=l)
  # fit/train the model
  model_fit = model.fit()
  coef = model_fit.params 
  history = train[len(train)-l:]
  history = [history[i] for i in range(len(history))]
  predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
  for t in range(len(test)):
    length = len(history)
    Lag = [history[i] for i in range(length-l,length)] 
    yhat = coef[0] # Initialize to w0
    for d in range(l):
      yhat += coef[d+1] * Lag[l-d-1] # Add other values 
    obs = test[t]
    predicted.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

  # computing rmse
  rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
  RMSE.append(rmse_per)

  # computing MAPE
  mape = np.mean(np.abs((test - predicted)/test))*100
  MAPE.append(mape)

# RMSE (%) and MAPE between predicted and original data values wrt lags in time sequence
data = {'Lag value':lag_val,'RMSE(%)':RMSE, 'MAPE' :MAPE}
print('Table 1\n',pd.DataFrame(data))

# plotting RMSE(%) vs. time lag
plt.xlabel('Time Lag')
plt.ylabel('RMSE(%)')
plt.title('RMSE(%) vs. time lag')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],RMSE)
plt.show()

# plotting MAPE vs. time lag
plt.xlabel('Time Lag')
plt.ylabel('MAPE')
plt.title('MAPE vs. time lag')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],MAPE)
plt.show()

# 4
print('4:')

# computing number of optimal value of p
p = 1
while p < len(cases):
  corr = pearsonr(train[p:].ravel(), train[:len(train)-p].ravel())
  if(abs(corr[0]) <= 2/math.sqrt(len(train[p:]))):
    print('The heuristic value for the optimal number of lags is',p-1)
    break
  p+=1

p=p-1
# training the model
model = AutoReg(train, lags=p)
# fit/train the model
model_fit = model.fit()
coef = model_fit.params 
history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
for t in range(len(test)):
  length = len(history)
  Lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0] # Initialize to w0
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] # Add other values 
  obs = test[t]
  predicted.append(yhat) #Append predictions to compute RMSE later
  history.append(obs) # Append actual test value to history, to be used in next step.

# computing rmse
rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
print('RMSE(%):',rmse_per)

# computing MAPE
mape = np.mean(np.abs((test - predicted)/test))*100
print('MAPE:',mape)

newmodel = AutoReg(X, lags=p)
# fit/train the model
newmodel_fit = newmodel.fit()
coef = newmodel_fit.params 
history = X[len(X)-p:]
history = [history[i] for i in range(len(history))]
predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
for t in range(len(test)):
  length = len(history)
  Lag = [history[i] for i in range(length-p,length)] 
  yhat = coef[0] # Initialize to w0
  for d in range(p):
    yhat += coef[d+1] * Lag[p-d-1] # Add other values 
  obs = test[t]
  predicted.append(yhat) #Append predictions to compute RMSE later
  history.append(obs) # Append actual test value to history, to be used in next step.