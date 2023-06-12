from os import stat
import numpy as np


class Metrics():
    def __init__(self):
        pass
    @staticmethod
    def sharpe_ratio(pnl):
        print('sharpe ratio: ' + str(pnl.mean() / pnl.std() * 252**0.5))
    @staticmethod
    def sum_pnl(pnl):
        print('PnL: ' + str(pnl.mean() * 252 * 100) + '%')
    @staticmethod
    def vol_pnl(pnl):
        print('Annual volatility: ' + str(pnl.std() * 252**0.5 * 100) + '%')
    @staticmethod
    def compute_variance(w, cov_matrix):
        return (w.T @ cov_matrix @ w)

class Portfolio():
    def __init__(self):
        pass
    @staticmethod
    def min_variance(cov_matrix):
        ones = np.ones((len(cov_matrix), 1))
        inv = np.linalg.inv(cov_matrix)
        w = inv @ ones
        w = w/np.abs(w).sum()
        return w
    @staticmethod
    def get_cov_matrix(data):
        rho_times_vol = data.set_index('asset_identifier').eval("rho * volatility")
        variance      = data.set_index('asset_identifier').eval("volatility**2").values

        rho_times_vol = rho_times_vol.values.reshape((-1, 1))
        cov_matrix = rho_times_vol @ rho_times_vol.T
    #     cov_matrix = np.zeros((len(data), len(data)))
        for i in range(len(cov_matrix)):
            cov_matrix[i][i] = variance[i]
        return cov_matrix

class RL():
    def __init__():
        pass
    @staticmethod
    def print_state(env):
        features = env._state_to_features(True)
        for key, value in features.items():
            print(key, value)

    @staticmethod
    def make_random_step(env, verbose = True):
        if verbose:
            RL.print_state(env)
        action = env.action_space.sample()
        if verbose:
            print(action)
        state, reward, done, _ = env.step(action)
        if verbose:
            RL.print_state(env)
            
    @staticmethod
    def make_model_step(env, model, verbose = True):
        obs                  = env.state
        action, _            = model.predict(obs, deterministic = True)
        obs, reward, done, _ = env.step(action)
        return env, [obs, reward, done]