# M4-time-series-forecasting
The dataset used was the M4 competition dataset. See more information on this in the wikepedia page : https://en.wikipedia.org/wiki/Makridakis_Competitions

This is the code develop to compare model for my undergraduate thesis in time series forecasting. The goal was to compare usual statistical method of forecasting to different state of the art neural network approach.
This code was written only to produce output and re-usability was not a goal as no work needs to be derived from those results. 

[![Capture-d-cran-2021-01-06-11-44-01.png](https://i.postimg.cc/NfLwrcCs/Capture-d-cran-2021-01-06-11-44-01.png)](https://postimg.cc/PPG7BBk9)

- main_analysis : This code outputs  metrics per model on the 4 selected series used in my bachelor thesis
- main : This code outputs metrics on all the time series selected
- Analysis_ARIMA : This code is focusing on the ARIMA models and is testing multiple parameters to optimize the ARIMA performance
- function : The code is composed of the useful function such as trend detection, stationarity test, data transformation
- models_bench : This code is used to produce the benchmark used in the competition 
- metrics : This code is the implementation of the metrics used in the competition
