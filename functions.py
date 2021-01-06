# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:22:12 2020

@author: darmo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from metrics import smape 


def stationary_test(ts):
    #We first log transform the ts to better represent linear trends
    
    ts = np.log(ts)
    result = adfuller(ts)
    p_value = result[1]
    stationary = None
    if p_value > 0.05:
        #rejection of null hypothesis (ts probably not stationary)
        stationary = False
    else:
        #We reject the null hypothesis, there is no unit root the ts is stationary
        stationary = True
        
    return stationary


def detrend(insample_data):
    """
    Calculates a & b parameters of LRL

    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices

    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)

    return si


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS

    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    """
    As noted by Professor Isidro Lloret Galiana:
    line 82:
    if len(ts_init) % 2 == 0:
    
    should be changed to
    if window % 2 == 0:
    
    This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
    In order for the results to be fully replicable this change is not incorporated into the code below
    """
    
    if len(ts_init) % 2 == 0:
         ts_ma = pd.Series(ts_init).rolling(window, center = True).mean()
         ts_ma = pd.Series(ts_ma).rolling(2, center = True).mean()
         ts_ma = np.roll(ts_ma, -1)
#        ts_ma = pd.rolling_mean(ts_init, window, center=True)
#        ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
#        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = pd.Series(ts_init).rolling(window, center = True).mean()

    return ts_ma


def seasonality_test(original_ts, ppy):
    """
    Seasonality test

    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    
    # Note that the statistical benchmarks, implemented in R, use the same seasonality test, but with ACF1 being squared
    # This difference between the two scripts was mentioned after the end of the competition and, therefore, no changes have been made 
    # to the existing code so that the results of the original submissions are reproducible
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (np.sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function

    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)

def pick_best_ES(a, b, c, y_test):
    smape1 = smape(a, y_test)
    smape2 = smape(b, y_test)
    smape3 = smape(c, y_test)
    
    if min(smape1, smape2, smape3) == smape1:
        return a
    if min(smape1, smape2, smape3) == smape2:
        return b
    if min(smape1, smape2, smape3) == smape3:
        return c
    
    


def find_loc_of_NaN(df):
    result = np.argwhere(np.isnan(df))
    if result.shape[0] != 0:
        return result[0][0]
    else:
        return len(df)
        
        
def split_into_train_test(data, in_num, fh, LSTM = False):
    """
    Splits the series into train and test sets. Each step takes multiple points as inputs

    :param data: an individual TS
    :param fh: number of out of sample points
    :param in_num: number of input points for the forecast
    :return:
    """
        
    train, test = data[:-fh], data[-(fh + in_num):]
    x_train, y_train = train[:-1], np.roll(train, -in_num)[:-in_num]
    x_test, y_test = train[-in_num:], np.roll(test, -in_num)[:-in_num]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = np.reshape(x_train, (-1, 1))
    x_test = np.reshape(x_test, (-1, 1))
    temp_test = np.roll(x_test, -1)
    temp_train = np.roll(x_train, -1)
    for x in range(1, in_num):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        x_test = np.concatenate((x_test[:-1], temp_test[:-1]), 1)
        temp_test = np.roll(temp_test, -1)[:-1]
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train, x_test, y_test


def save_forecasted_values(x_test, y_test, forecast, name = None):
    plt.clf()
    ####REPLACE NEXT LINE WITH THE COMMENTED ONE IF RUNNED WILL MULTIPLE TS####
    #x_test = x_test
    x_test = x_test[0]
    first_part = range(1, len(x_test) + 1)
    second_part = range(len(x_test), len(x_test) + len(y_test))
    
    plt.plot(first_part, x_test, label = 'data', color = 'blue')
    plt.plot(second_part, y_test, color = 'blue')
    plt.plot(second_part, forecast, color = 'red', label = 'forecast')
    
    #Connect x_test with y_test
    plt.plot([len(x_test), len(x_test)], [x_test[-1], y_test[0]], color = 'blue')
    #connect x_test with forecast
    plt.plot([len(x_test), len(x_test)], [x_test[-1], forecast[0]], color = 'red')
    plt.legend(loc='best', fontsize='xx-large')
    number = np.random.randint(0, 999999999)
    if name != None:
        plt.savefig(f'image/{name}{number}.png')
    else:
        plt.savefig(f'image/plot.png')

def compute_all_acf_plots(data, lag = 30):
    counter = 0 
    for ts in data:
        ts = ts[~np.isnan(ts)]
        plot_acf(ts, lags=lag)
        plt.savefig(f'image/acf{counter}.png')
        counter += 1
        
def compute_all_pacf_plots(data, lag = 30):
    counter = 0 
    for ts in data:
        ts = ts[~np.isnan(ts)]
        plot_pacf(ts, lags=lag)
        plt.savefig(f'image/pacf{counter}.png')
        counter += 1
        
        
def delete_image_folder():
    try:
        shutil.rmtree('image')
        print('image folder deleted')
    except OSError as e:
        print("Error: %s : %s" % ('image', e.strerror))

def create_image_folder():
    os.mkdir('image')
    print('image folder created')
    

    