#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:23:52 2020

@author: tomdarmon
"""

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd
from functions import stationary_test, delete_image_folder, create_image_folder, compute_all_acf_plots, save_forecasted_values
import matplotlib.pyplot as plt
import os
from models_bench import *
from metrics import *


#reset the image folder, not working if called from an imported script 
if os.path.isdir('image'):
    delete_image_folder()
    create_image_folder()
else:
    create_image_folder()

data_all = pd.read_csv(f'/Users/tomdarmon/Documents/Thesis Bocconi/project/train/Daily-train.csv')
data_all = data_all.drop('V1', axis = 1)
data_all = data_all.values

fh = 14
freq = 30

'''
i = 0

### LOOKING FOR INTERESTING PLOTS ####

for ts in data_all:
    plt.plot(ts)
    plt.savefig(f'image/time_plots{i}')
    i += 1
    plt.clf()
    
### INTERESTING AT 379(seasonal), 1437 (maybe mult seasonality), 4130 (seasonal), 2131
'''
ts_all = [data_all[379],data_all[1437],data_all[4130],data_all[2131]]
for i in range(len(ts_all)):
    ts_all[i] = ts_all[i][~np.isnan(ts_all[i])]
    

stationary = [stationary_test(ts) for ts in ts_all]
#### stationary = [F, F, T, F] but the first ts look stationary, the test has a p value of 0.06
#so it is rejected with a small margin, we can assume it is because the ts stop at the bottom. We will assume
#the first ts is stationary
stationary[0] = True

MAPE = list()
MASE = list()
y_hats_ARIMA = list()

for i in range(len(ts_all)):
    y_hat_ARIMAs = list()
    if stationary[i] == False:
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (0,1,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (1,1,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (2,1,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (3,1,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (4,1,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (0,1,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (1,1,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (3,1,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (4,1,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (5,1,1)))
        
    else:
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (2,0,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (3,0,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (4,0,0)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (0,0,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (1,0,1))) 
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (2,0,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (3,0,1)))
        y_hat_ARIMAs.append(arima_bench(ts_all[i][:-fh], fh, o = (4,0,1)))
    
        #y_test is the last 14 value (that was previsously taken out of the ts in: arima_bench(ts[i][:-fh] )
    y_test = ts_all[i][-14:]
    x_test = ts_all[i][:-fh]
    all_mase = [mase(ts_all[i][:-fh], y_test, y_hat_ARIMAs[j], freq) for j in range(len(y_hat_ARIMAs))]
        
    final_mase = np.min(all_mase)
    min_index = np.argmin(all_mase)
    y_hat_ARIMA = y_hat_ARIMAs[min_index]
    final_mape = smape(y_test, y_hat_ARIMA)
    
    MAPE.append(final_mape)
    MASE.append(final_mase)
    y_hats_ARIMA.append(y_hat_ARIMA)
    
    save_forecasted_values(x_test[-200:], y_test, y_hat_ARIMA, name = 'ARIMA')
    
            
    
    
    















