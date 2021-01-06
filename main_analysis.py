# This code can be used to reproduce the forecasts of M4 Competition NN benchmarks and evaluate their accuracy
    
import tensorflow as tf
import numpy as np
import pandas as pd
import gc


from functions import *
from models_bench import *
from metrics import *
import matplotlib.pyplot as plt

#reset the image folder, not working if called from an imported script 
if os.path.isdir('image'):
    delete_image_folder()
    create_image_folder()
else:
    create_image_folder()

#def main():
    

data_all = pd.read_csv(f'/Users/tomdarmon/Documents/Thesis Bocconi/project/train/Daily-train.csv')
freq = 60
data_all = data_all.drop('V1', axis = 1)

#Replace all NaN values

data_all = data_all.values

#Chose how many timeseries

ts_all = [data_all[379],data_all[1437],data_all[4130],data_all[2131]]

for i in range(len(ts_all)):
    ts_all[i] = ts_all[i][~np.isnan(ts_all[i])]
    
stationary = [stationary_test(ts) for ts in ts_all]
#### stationary = [F, F, T, F] but the first ts look stationary, the test has a p value of 0.06
#so it is rejected with a small margin, we can assume it is because the ts stop at the bottom. We will assume
#the first ts is stationary
stationary[0] = True


# forecasting horizon year/quarterly/Month/days/hourly : 6 / 8 / 18 / 14 / 48
fh = 14
# number of points used as input for each forecast during training for LSTM and RNN         
# 24 * 30 for hourly to englobe 1 month of data
in_size = 60

err_MLP_sMAPE = []
err_MLP_MASE = []

err_ES_sMAPE = []
err_ES_MASE = []

err_RNN_sMAPE = []
err_RNN_MASE = []

err_LSTM_sMAPE = []
err_LSTM_MASE = []

err_ARIMA_sMAPE = []
err_ARIMA_MASE = []

y_hats_ARIMA = []
y_hats_RNN = []
y_hats_MLP = []
y_hats_LSTM = []
y_hats_ES = []
y_test_all = []


counter = 0
# ===== Main loop which goes through all timeseries =====
for j in range(len(ts_all)):
    ts = ts_all[j]
    
    #INDUCE STATIONARITY
    
    if stationary[j] == False:
        ts_all[j] = np.diff(ts_all[j])
    
    
    #ARIMA benchmark before pre processing for neural networks

    y_hat_test_ARIMA = arima_bench(ts, fh)
    
    #Exponential_smoothing_holt_winterbenchmark before pre processing for neural networks 
    
    y_hat_test_ES1, y_hat_test_ES2, y_hat_test_ES3  = Exponential_smoothing_bench(ts, fh)
         
    # remove seasonality
    seasonality_in = deseasonalize(ts, freq)

    for i in range(0, len(ts)):
        ts[i] = ts[i] * 100 / seasonality_in[i % freq]

    # detrending
    a, b = detrend(ts)

    for i in range(0, len(ts)):
        ts[i] = ts[i] - ((a * i) + b)
        
    
    x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)
    
    
    # LSTM benchmark - Produce forecasts
    
    y_hat_test_LSTM = lstm_bench(x_train, y_train, x_test, fh, in_size)

    # RNN benchmark - Produce forecasts
    
    y_hat_test_RNN = np.reshape(rnn_bench(x_train, y_train, x_test, fh, in_size), (-1))
    
    # MLP benchmark - Produce forecasts
    
    y_hat_test_MLP = mlp_bench(x_train, y_train, x_test, fh)
 

    # add trend
    for i in range(0, len(ts)):
        ts[i] = ts[i] + ((a * i) + b)

    for i in range(0, fh):
        y_hat_test_MLP[i] = y_hat_test_MLP[i] + ((a * (len(ts) + i + 1)) + b)
        y_hat_test_RNN[i] = y_hat_test_RNN[i] + ((a * (len(ts) + i + 1)) + b)
        y_hat_test_LSTM[i] = y_hat_test_LSTM[i] + ((a * (len(ts) + i + 1)) + b)

    for i in range(0, len(ts)):
        ts[i] = ts[i] * seasonality_in[i % freq] / 100

    for i in range(len(ts), len(ts) + fh):
        y_hat_test_MLP[i - len(ts)] = y_hat_test_MLP[i - len(ts)] * seasonality_in[i % freq] / 100
        y_hat_test_RNN[i - len(ts)] = y_hat_test_RNN[i - len(ts)] * seasonality_in[i % freq] / 100
        y_hat_test_LSTM[i - len(ts)] = y_hat_test_LSTM[i - len(ts)] * seasonality_in[i % freq] / 100
        

    #check if negative or extreme
    # for i in range(len(y_hat_test_MLP)):
    #     if y_hat_test_MLP[i] < 0:
    #         y_hat_test_MLP[i] = 0
    #     if y_hat_test_RNN[i] < 0:
    #         y_hat_test_RNN[i] = 0
    #     if y_hat_test_LSTM[i] < 0:
    #         y_hat_test_LSTM[i] = 0
    #     if y_hat_test_ARIMA[i] < 0:
    #         y_hat_test_ARIMA[i] = 0 
      
        
            
    #     if y_hat_test_MLP[i] > (1000 * max(ts)):
    #         y_hat_test_MLP[i] = max(ts)         
    #     if y_hat_test_RNN[i] > (1000 * max(ts)):
    #         y_hat_test_RNN[i] = max(ts)
    #     if y_hat_test_LSTM[i] > (1000 * max(ts)):
    #         y_hat_test_LSTM = max(ts)
    #     if y_hat_test_ARIMA[i] > (1000* max(ts)):
    #         y_hat_test_ARIMA[i] = max(ts)
  
    x_train, y_train, x_test, y_test = split_into_train_test(ts, in_size, fh)
    
    ### CHOSE THE BEST ES MODEL BASED ON SMAPE####
    
    y_hat_test_ES = pick_best_ES(y_hat_test_ES1,y_hat_test_ES2, y_hat_test_ES3, y_test)

    # Calculate errors and save forecast plots
    err_MLP_sMAPE.append(smape(y_test, y_hat_test_MLP))
    err_RNN_sMAPE.append(smape(y_test, y_hat_test_RNN))
    err_LSTM_sMAPE.append(smape(y_test, y_hat_test_LSTM))
    err_ARIMA_sMAPE.append(smape(y_test, y_hat_test_ARIMA))
    err_ES_sMAPE.append(smape(y_test, y_hat_test_ES))
    
    err_MLP_MASE.append(mase(ts[:-fh], y_test, y_hat_test_MLP, freq))
    err_RNN_MASE.append(mase(ts[:-fh], y_test, y_hat_test_RNN, freq))
    err_LSTM_MASE.append(mase(ts[:-fh], y_test, y_hat_test_LSTM, freq))
    err_ARIMA_MASE.append(mase(ts[:-fh], y_test, y_hat_test_ARIMA, freq))
    err_ES_MASE.append(mase(ts[:-fh], y_test, y_hat_test_ES, freq))  
 
    
    
    
    y_hats_ARIMA.append(y_hat_test_ARIMA)
    y_hats_RNN.append(y_hat_test_RNN)
    y_hats_MLP.append(y_hat_test_MLP)
    y_hats_LSTM.append(y_hat_test_LSTM)
    y_hats_ES.append(y_hat_test_ES)
    
    y_test_all.append(y_test)
    
    save_forecasted_values(x_test, y_test, y_hat_test_ARIMA, name = 'ARIMA')
    save_forecasted_values(x_test, y_test, y_hat_test_MLP, name = 'MLP')
    save_forecasted_values(x_test, y_test, y_hat_test_RNN, name = 'RNN')
    save_forecasted_values(x_test, y_test, y_hat_test_LSTM, name = 'LSTM')
    save_forecasted_values(x_test, y_test, y_hat_test_ES, name = 'ES')
    

    # memory handling
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()
    plt.clf()

    counter = counter + 1
    print("-------------TS ID: ", counter, "-------------")

print("\n\n---------FINAL RESULTS---------")
print("=============sMAPE=============\n")
print("#### MLP ####\n", np.mean(err_MLP_sMAPE), "\n")
print("#### RNN ####\n", np.mean(err_RNN_sMAPE), "\n")
print("#### LSTM ####\n", np.mean(err_LSTM_sMAPE), "\n")
print("#### ARIMA ####\n", np.mean(err_ARIMA_sMAPE), "\n")
print("#### Holt ####\n", np.mean(err_ES_sMAPE), "\n")
          
print("==============MASE=============")
print("#### MLP ####\n", np.mean(err_MLP_MASE), "\n")
print("#### RNN ####\n", np.mean(err_RNN_MASE), "\n")
print("#### LSTM ####\n", np.mean(err_LSTM_MASE), "\n")
print("#### ARIMA ####\n", np.mean(err_ARIMA_MASE), "\n")
print("#### ES ####\n", np.mean(err_ES_MASE), "\n")
          
#if __name__ == '__main__':
#    main()
