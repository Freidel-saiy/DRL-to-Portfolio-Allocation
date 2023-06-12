import pandas as pd
import numpy as np
import os

from lib.RL_portfolio.pipeline import Pipeline

### PATH PARAMETERS ################################################
#Read inputs
path_data      = str(input("Pass the folder with the data to be processed: "))
path_stocks    = os.path.join(path_data, 'df_stocks.pkl')
path_benchmark = os.path.join(path_data, 'df_benchmark.pkl')
while not (os.path.exists(path_stocks) & os.path.exists(path_benchmark)):
    print("in the folder that you pass, there should be a df_stocks.pkl and a df_benchmark.pkl")
    path_data      = str(input("Pass the folder with the data to be processed: "))
    path_stocks    = os.path.join(path_data, 'df_stocks.pkl')
    path_benchmark = os.path.join(path_data, 'df_benchmark.pkl')

### MODELING PARAMETERS ##########################################
#Some hard coded parameters of the process
data_parameters = {'n_steps': 60}
date_parameters = dict(train_start_date = '2004-01-01',
                       gap_train_test   = 5,
                       test_start_date  = '2010-01-01')
lstm_parameters     = dict(units = 20, activation='relu')
optmizer_parameters = dict(optimizer='adam', loss='mse')
fit_parameters      = dict(epochs=100, verbose=0, batch_size = 1024)
model_parameters    = {'lstm_parameters': lstm_parameters,
                       'optmizer_parameters': optmizer_parameters,
                       'fit_parameters': fit_parameters}
risk_parameters = {'vol_window': 20, 'corr_window': 60}

### PROCESS DATA #############################################
#Read
df_stocks    = pd.read_pickle(path_stocks)
df_benchmark = pd.read_pickle(path_benchmark)
#Make predictions about the future
pipeline = Pipeline()
data_train, data_test = pipeline.make_predictions(df_stocks, data_parameters, date_parameters, model_parameters)
#Compute risk variables
data_test = pipeline.compute_risk_variables(data_test, df_benchmark, risk_parameters)

### SAVE DATA #############################################
path_save = os.path.join(path_data, 'df_stocks_variables.pkl')
data_test.to_pickle(path_save)