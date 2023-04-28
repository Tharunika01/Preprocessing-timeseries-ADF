from pandas import Series
import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from numpy import log
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools
import statsmodels.api as sm
import math
#Import the dataset directly as a series and make the index cloumn 0
AirPassengers = pd.read_csv('AirPassengers.csv',header=0, index_col=0, parse_dates=True)
#read the data it as pandas dataframe 
# load data set and save to DataFrame
temp = pd.read_csv('22_daily-minimum-temperatures-in-me.csv', header=0, index_col=0, parse_dates=True)
# create Series object
temperatures_series = temp['Daily minimum temperatures in Melbourne']
#checking null values
print ('Dataset contain null:\t',AirPassengers.isnull().values.any())
print ('Dataset contain null:\t',temp.isnull().values.any())
#Print out the first 5 rows
print(AirPassengers.head())
AirPassengers.plot()
plt.title('Number of Airline passengers vs Date')
plt.ylabel('Number of Airline passengers ')
plt.xlabel('Date')
pyplot.show()
print(temperatures_series.head())
temperatures_series.plot()
plt.title('Daily minimum temperatures in Melbourne vs Date')
plt.ylabel('Daily minimum temperatures in Melbourne')
plt.xlabel('Date')
pyplot.show()
plot_pacf(AirPassengers)
# fit model
AirPassengers = AirPassengers.astype(float)
model = sm.tsa.arima.ARIMA(AirPassengers, order=(5,1,5))
model_fit = model.fit()
print(model_fit.summary())
from pandas import DataFrame
residuls = DataFrame(model_fit.resid)
model_fit.resid
residuls.plot()
pyplot.show()
residuls.plot(kind='kde')
pyplot.show()
a=residuls.describe()
print(a)
print('\n')
print('Auto correlation')
autocorrelation_plot(temperatures_series)
pyplot.show()
autocorrelation_plot(AirPassengers)
pyplot.show()
#Divide the dataset to 80% train set and 20% test set
#AirPassengers_series = AirPassengers_series.astype(float)
size = int(len(AirPassengers) * 0.80)
train_set, test_set = AirPassengers[0:size], AirPassengers[size:len(AirPassengers)]
history = [AirPassengers for AirPassengers in train_set]
predictions = list()
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
# specify to ignore warning messages
warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_set,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('SARIMAX{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
# fiiting the arima model
mod = sm.tsa.statespace.SARIMAX(train_set,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 15))
plt.show()
# forecasting from 1st January 1958
# dynamic=False argument ensures that we produce one-step ahead forecasts
predicted = results.get_prediction(start=pd.to_datetime('1955-01-01'), dynamic=False)
predicted_conf = predicted.conf_int()
ax = AirPassengers['1950':].plot(label='observed')
predicted.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(predicted_conf.index,
                predicted_conf.iloc[:, 0],
                predicted_conf.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
plt.legend()
plt.show()
AirPassengers_forecasted = predicted.predicted_mean
AirPassengers_truth = AirPassengers['1955-01-01':]
# Compute the mean square error
mse = ((AirPassengers_forecasted - AirPassengers_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#print('The Root Mean Squared Error of our forecasts is {}'.format( math.sqrt(mse)))
#dynamic forecasting
pred_dynamic = results.get_prediction(start=pd.to_datetime('1955-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
ax = AirPassengers['1950':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1958-01-01'), AirPassengers.index[-1],
                 alpha=.1, zorder=-1)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
plt.legend()
plt.show()
# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = AirPassengers['1955-01-01':]
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
ax = AirPassengers.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
plt.legend()
plt.show()
##Using the test set 
#for t in range(len(test_set)):
#    model = ARIMA(train_set, order=(5,1,1))
#    model_fit = model.fit(disp=0)
#    output = model_fit.forecast()
#    predicted_value = output[0]
#    predictions.append(predicted_value)
#    expected_value = test_set[0]
#    history.append(expected_value)
#    error = expected_value - predicted_value
#    print('predicted of Y = %f, expected of Y = %f and error = %f' % (predicted_value, expected_value, error))
#final_error = mean_squared_error(test_set, predictions)
#print('Test Mean Sqaured Error: %.3f' % final_error)
#print('\n')
#replace ment
mymodel = sm.tsa.arima.ARIMA(train_set, order = (1, 1, 2))  
modelfit = mymodel.fit()  
print(modelfit.summary())  
mymodel = sm.tsa.arima.ARIMA(train_set, order = (1, 1, 1))  
modelfit = mymodel.fit()
# Plotting Residual Errors  
myresiduals = pd.DataFrame(modelfit.resid)  
fig, ax = plt.subplots(1,2)  
myresiduals.plot(title = "Residuals", ax = ax[0])  
myresiduals.plot(kind = 'kde', title = 'Density', ax = ax[1])  
plt.show()
#check if the mean and the moving variance varies with time 
#yearly moving averages (12 months window)
#window is Size of the moving window. This is the number of observations
#used for calculating the statistic.
print("Augmented Dicky Filler")
print('\n')
def stationary_check(dataset, lags=None): 
    rol_mean =dataset.rolling(12).mean()  #pd.rolling_mean(y, window=12)
    rol_std = dataset.rolling(12).std() #pd.rolling_std(y, window=12)
    #Plot 
    Original_dataset = plt.plot(dataset, color='blue',label='Original')
    mean = plt.plot(rol_mean, color='red', label='12-Months Rolling Mean')
    std = plt.plot(rol_std, color='black', label = '12-Months Rolling Std')
    plt.legend(loc='best')
    plt.title('Time series dataset')
    plt.show(block=False)
#lags to specify just the number of values to display
stationary_check(AirPassengers, lags=30)
#lags to specify just the number of values to display
stationary_check(temperatures_series, lags=30)
AirPassengers.hist()
pyplot.show()
#split the dataset into two and check for difference in mean and variance 
X = AirPassengers.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
temperatures_series.hist()
pyplot.show()
#future extraction
#split the dataset into two and check for difference in mean and variance
print("mean and varience")
X = temperatures_series.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
print("Making Air Passengers Dataset stationary")
#log transform can be used to flatten out exponential change back to a linear relationship.
AirPassengers_log = log(AirPassengers)
pyplot.hist(AirPassengers_log)
pyplot.show()
pyplot.plot(AirPassengers_log)
pyplot.show()