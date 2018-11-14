# load required libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# stationarity test
from statsmodels.tsa.stattools import adfuller
# calculate and plot the autocorrelation functions
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# functions to parse and plot time series
# using statsmodels library

def split_and_plot(timeseries, split_type= "additive", add_observed= True):

    ''' Decomposition of timeseries into trend, seasonality,
        and residuals. Plots each component, including
        time series observations. '''

    ts_decomp = seasonal_decompose(timeseries, model= split_type)

    d = { "Observed vs. Trend": (ts_decomp.trend, "orange", 0),
          "Seasonal Component": (ts_decomp.seasonal, "black", 1),
          "Residuals": (ts_decomp.resid, "red", 2) }

    fig, axes = plt.subplots(3, 1, figsize= (14, 8))
    plt.subplots_adjust(hspace= 1.0)

    for key, value in d.items():

        plt.subplot(3, 1, value[2]+1)

        if add_observed and key=="Observed vs. Trend":
            axes[value[2]] = plt.plot(timeseries, c= "grey")
            axes[value[2]] = plt.plot(value[0], c= value[1])
            axes[value[2]] = plt.title(key)
        else:
            axes[value[2]] = plt.plot(value[0], c= value[1])
            axes[value[2]] = plt.title(key)


def test_df(timeseries):

    dftest = adfuller(timeseries, autolag= "AIC")

    dfout = pd.Series(dftest[0:4], index= ["Test Statistic",
                                           "p-value",
                                           "#Lags Used",
                                           "#Observations Used"])

    return( ("Results of Dickey-Fuller Test:", dfout) )


def plot_autocorr(timeseries, nlags):

    ''' Time series autocorrelation and partial
        autocorrelation function, and plotting. '''

    timeseries.dropna(inplace= True)

    fig, axes = plt.subplots(1, 2, figsize= (12, 4))
    plt.subplots_adjust(wspace= 0.5)

    lag_acf = acf(timeseries, nlags= nlags)
    lag_pacf = pacf(timeseries, nlags= nlags, method= "ols")

    fig = plot_acf(timeseries, lags= nlags, ax= axes[0])
    fig = plot_pacf(timeseries, lags= nlags, ax= axes[1])


def plot_model(start, end, timeseries, predicted_values,
               predicted_ci= None, actual_values= None,
               title= ""):

    fig, axes = plt.subplots(1, 1, figsize= (12, 4))
    # plot timeseries
    ax = sns.lineplot(data= timeseries)
    # plot actual values
    if actual_values is not None:
        ax = sns.lineplot(data= actual_values[start:end],
                          color= "blue", linewidth= 0.75)
        ax.lines[1].set_linestyle("-.")

    # plot forecasts
    ax= sns.lineplot(data= predicted_values,
                     color= "red", linewidth= 1.0)

    # plot confidence intervals
    if predicted_ci is not None:
        y1= predicted_ci.iloc[:, 0].where(cond= predicted_ci.iloc[:, 0]>0,
                                          other= 0)
        y2= predicted_ci.iloc[:, 1]
        x= predicted_values.index
        ax.fill_between(x= x, y1= y1, y2= y2,
                        alpha=.15, color= "red")
    # add title
    ax= plt.title(title)
