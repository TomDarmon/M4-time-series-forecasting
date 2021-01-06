# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:26:43 2020

@author: darmo
"""

import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as ker
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
import numpy as np
import pandas as pd
import seaborn as sns

# implementation of auto_arima from R
#Some dependencies were deprecated but with importation renaming, all went well

'''def arima_bench_auto(ts, fh):
    ts = ts.flatten()
    model = auto_arima(ts, stationary = False, trend = True, maxiter= 100)
    model_fit= model.fit()
    y_hat_test = model_fit.forecast(fh)
    return y_hat_test'''

def Exponential_smoothing_bench(ts, fh):
    """
    Forecasts using ES models and keeping the best one

    :param ts: time serie
    :param fh: forecasting horizon
    :return:
    """
    ts = ts.flatten()
#    ts = np.log(ts)
    
    
    #### FIRST MODEL  : SIMPLE ES (no trend no seasonality) #####
    model1 = SimpleExpSmoothing(ts)
    model_fit1 = model1.fit(optimized = True)
    y_hat_test1 = model_fit1.forecast(fh)
    #### SECON MODEL  : DOUBLE ES (trend but no seasonality) #####
    model2 = Holt(ts)
    model_fit2 = model2.fit()
    y_hat_test2 = model_fit2.forecast(fh)
    #### THIRD MODEL  : HOLT WINTER ES (trend seasonality) #####
    model3 = ExponentialSmoothing(ts)
    model_fit3 = model3.fit()
    y_hat_test3 = model_fit3.forecast(fh)    
    

    #    y_hat_test = np.exp(y_hat_test)
    return y_hat_test1, y_hat_test2, y_hat_test3
    

def Exponential_smoothing_bench_holt(ts, fh):
    """
    Forecasts using ES model

    :param ts: time serie
    :param fh: forecasting horizon
    :return:
    """
    ts = ts.flatten()
#    ts = np.log(ts)
    model = Holt(ts)
    model_fit = model.fit()
    
    # make predictions
    
    y_hat_test = model_fit.forecast(fh)
    y_hat_test = np.exp(y_hat_test)
    return np.asarray(y_hat_test)
    

def arima_bench(ts, fh, o = (5,1,1)):
    """
    Forecasts using ARIMA model

    :param ts: time serie
    :param fh: forecasting horizon
    :return:
    """
#   the order still needs to be optimized
    ts = ts.flatten()
    model = ARIMA(ts, order = o)
    model_fit = model.fit()
    
    # make predictions
    
#    y_hat_test = model_fit.predict(start = len(ts), end = len(ts) + fh - 1)
    y_hat_test = model_fit.forecast(fh)[0]
#    sns.kdeplot(np.array(model_fit.resid), bw=0.5)
#    for i in range(0, fh):
#        y_hat_test.append(last_prediction)
#        x_test[0] = np.roll(x_test[0], -1)
#        x_test[0, (len(x_test[0]) - 1)] = last_prediction
#        last_prediction = model_fit.forecast(x_test)[0][0]

    return np.asarray(y_hat_test)


def rnn_bench(x_train, y_train, x_test, fh, input_size):
    """
    Forecasts using 6 SimpleRNN nodes in the hidden layer and a Dense output layer

    :param x_train: train data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :param input_size: number of points used as input
    :return:
    """
    # reshape to match expected input
    x_train = np.reshape(x_train, (-1, input_size, 1))
    x_test = np.reshape(x_test, (-1, input_size, 1))

    # create the model
    model = Sequential([
        SimpleRNN(3, input_shape=(input_size, 1), activation='linear',
                  use_bias=False, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros',
                  dropout=0.0, recurrent_dropout=0.0),
        Dense(1, use_bias=True, activation='linear')
    ])
    opt = RMSprop(lr=0.005)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # fit the model to the training data
    model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=1)

    # make predictions
    y_hat_test = []
    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)



def mlp_bench(x_train, y_train, x_test, fh):
    """
    Forecasts using a simple MLP which 6 nodes in the hidden layer

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :return:
    """
    y_hat_test = []

    model = MLPRegressor(hidden_layer_sizes=8, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.01,
                         random_state=42)
    model.fit(x_train, y_train)

    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)

def lstm_bench(x_train, y_train, x_test, fh, input_size, n_features = 1):
    """
    Forecasts using a RNN which 3 hidden layers
    
    the dimension of the input is in 3D of the form : 
    reshape from [samples, timesteps] into [samples, timesteps, features

    :param x_train: train input data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :param n_features: number of variables (if univariate = 1)
    :return:
        
    """
#    the dimension of the input is in 3D of the form : 
#    reshape from [samples, timesteps] into [samples, timesteps, features
                 
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], n_features))
    x_test = np.reshape(x_test, (1,input_size, n_features))
    
    model = Sequential()
    model.add(LSTM(16, activation='relu', return_sequences = True))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer = RMSprop(lr = 0.0005), loss = 'mse')
    
    model.fit(x_train, y_train, epochs = 4, verbose = 1)
    
    # make predictions
    y_hat_test = []
    last_prediction = model.predict(x_test)[0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0, (len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0]

    return np.asarray(y_hat_test)
    



