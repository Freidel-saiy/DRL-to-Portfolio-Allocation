from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.random import set_seed
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# get the model
def get_model(n_inputs, n_outputs, optimizer_params):
    model = Sequential()
    model.add(Dense(64, input_dim=n_inputs, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(**optimizer_params)
    return model

def eval_model(X, y, fit_params = {}, optimizer_params = {}, seed = 0):
    set_seed(seed)
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    model = get_model(n_inputs, n_outputs, optimizer_params)

    hist = model.fit(X, y, **fit_params)
    plt.plot(hist.history['loss'])
    yhat = model.predict(X)

    correlation = np.corrcoef(yhat.reshape(-1), y.reshape(-1))
    yhat_rescaled = LinearRegression().fit(yhat, y).predict(yhat)
    
    results = {'correlation': [correlation[0, 1]], 'r2': [r2_score(y, yhat_rescaled)]}

    return results

def pnl_metric(y, yhat):
    notional = yhat - np.median(yhat)
    notional = notional / np.sum(np.abs(notional))

    pnl = np.sum(y * notional)
    return pnl

def pnl_curve(y, yhat, dates):
    notional = yhat - np.median(yhat)
    notional = notional / np.sum(np.abs(notional))
    
    df_performance = pd.DataFrame({"notional": notional,
                                    "returns": y,
                                    "date": dates})
    df_performance.eval("pnl = notional * returns", inplace = True)
    pnl = df_performance.groupby('date')['pnl'].sum()
    return pnl

def sharpe_metric(y, yhat, dates):
    notional = yhat - np.median(yhat)
    notional = notional / np.sum(np.abs(notional))
    
    df_performance = pd.DataFrame({"notional": notional,
                                    "returns": y,
                                    "date": dates})
    df_performance.eval("pnl = notional * returns", inplace = True)
    pnl = df_performance.groupby('date')['pnl'].sum()
    annualized_sharpe = pnl.mean() / pnl.std() * 252**0.5
    return annualized_sharpe