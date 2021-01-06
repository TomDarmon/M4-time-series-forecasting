# M4-time-series-forecasting
This is the code develop to compare model for my undergraduate thesis in time series forecasting. The goal was to compare usual statistical method of forecasting to different state of the art neural network approach.
This code was written only to produce output and re-usability was not a goal as no work needs to be derived from those results. 

- main_analysis : This code outputs  metrics per model on the 4 selected series used in the thesis
- main : This code outputs metrics on all the time series selected
- Analysis_ARIMA : This code is focusing on the ARIMA models and is testing multiple parameters to optimize the ARIMA performance
- function : The code is composed of the useful function such as trend detection, stationarity test, data transformation
- models_bench : This code is used to produce the benchmark used in the competition 
- metrics : This code is the implementation of the metrics used in the competition
