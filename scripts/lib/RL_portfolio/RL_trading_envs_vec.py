from copy import deepcopy
from matplotlib.style import available
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import math
import time

class AuxiliarEnvMethods():
    def __init__(self):
        pass

    def _set_debug_lists(self):
        self.hidden_state_memory = {}
        self.action_memory = []
        self.state_memory = []
        self.hidden_state_before_action_memory = []
        self.hidden_state_after_action_memory = []

    def _relu(self, value, base_value):
        if value > base_value:
            return value - base_value
        else:
            return 0
    
    def _set_hidden_state(self):
        #Each key of the hidden state will be one asset
        #Each value will be a list containing: current notional, current volatility, last action, current correlation
        self.hidden_state = {}
        date = self.df.loc[self.rl_timestamp, 'date']
        data = self.df.query("date == @date")
        assets = list(self.df['asset_identifier'].unique())
        for asset in assets:
            data_asset = data.query("asset_identifier == @asset")
            if not data_asset.empty:
                self.hidden_state[asset] = [0, data_asset['volatility'].item(), np.nan, data_asset['rho'].item()]
            else:
                self.hidden_state[asset] = [0, np.nan, np.nan, np.nan]

    def _update_hidden_state_new_day(self):
        portfolio_volatility = 0
        portfolio_corr = 0
        available_notional = 1
        #Loop through stocks to compute (updating) the portfolio variables
        #and the hidden state
        for stock, values in self.hidden_state.items():
            #Get new information for the given stock
            date = self.v_date[self.rl_timestamp]
            data = self.d_date_stock[date].get(stock)
            if data is None: #Wont interfer with the available notional. Also, its notional will become zero
                #Update the hidden state
                self.hidden_state[stock] = [0, np.nan, np.nan, np.nan]

            else:
                realized_return = data['realized_return']
                new_vol         = data['volatility']
                new_corr        = data['rho']
                old_notional    = values[0]
                new_notional    = old_notional * (1 + realized_return)
                #Update the hidden state
                self.hidden_state[stock] = [new_notional, new_vol, np.nan, new_corr]
                if new_notional != 0:
                    #Update portfolio variables
                    initial_portfolio_volatility = portfolio_volatility
                    portfolio_volatility, portfolio_corr = self._update_portfolio_volatility(new_notional, portfolio_volatility, new_vol, 0, new_corr, portfolio_corr)
                    available_notional -= np.abs(new_notional)
        return portfolio_volatility, portfolio_corr, available_notional

    def _reorder_df_by_absolute_notional(self):
        #Get notional absolute values to order by
        notional_dict = {asset: np.abs(values[0]) for asset, values in self.hidden_state.items()}
        #Separete data for a given day and sort it
        date = self.df.loc[self.rl_timestamp, 'date']
        date_mask = self.df['date'] == date
        data = self.df.loc[date_mask, :].copy()
        index = data.index
        data = data.sort_values('asset_identifier',
                                key = lambda x: x.map(notional_dict),
                                ascending = False).set_index(index)
        #Replace in the dataframe
        self.df.loc[date_mask, :] = data

    def _update_hidden_state_notional(self, asset, notional, allocation):
        self.hidden_state[asset][0] = notional
        self.hidden_state[asset][2] = allocation

    def _get_current_notional(self, asset):
        return self.hidden_state[asset][0]

    def _bound_allocation(self, allocation, available_notional, initial_stock_notional):
        initial_available_notional = available_notional
        if (available_notional <= 0) and (allocation * initial_stock_notional > 0):#There is no available notional and you are trying to increase position
            stock_notional = initial_stock_notional
        else:
            available_notional = available_notional + np.abs(initial_stock_notional) #Take off the asset from the portfolio
            stock_notional     = initial_stock_notional + allocation #Calculate the desired new notional
            #Clip this new notional
            if available_notional > 0:
                stock_notional = min(available_notional, max(stock_notional, -available_notional)) 
            elif allocation * initial_stock_notional > 0: #There is no available notional and you are trying to increase position
                stock_notional = initial_stock_notional
            else: #There is no available notional and you are trying to reduce your position
                if initial_stock_notional > 0:
                    stock_notional = max(0, stock_notional)
                elif initial_stock_notional < 0:
                    stock_notional = min(0, stock_notional)
                else:
                    stock_notional = 0
        available_notional = initial_available_notional + np.abs(initial_stock_notional) - np.abs(stock_notional)
        allocation         = stock_notional - initial_stock_notional
        return allocation, available_notional, stock_notional

    def _state_to_features(self, as_dict = False):
        portfolio_volatility = self.state[0] / 100
        stock_volatility     = self.state[1] / 100
        stock_prediction     = self.state[2] / 1e4
        available_notional   = self.state[3]
        stock_notional       = self.state[4] / 10
        stock_corr           = self.state[5] / 10
        portfolio_corr       = self.state[6] / 10
        if as_dict:
            return {'portfolio_volatility': portfolio_volatility,
                   'stock_volatility': stock_volatility,
                   'stock_prediction': stock_prediction,
                   'available_notional': available_notional,
                   'stock_notional': stock_notional,
                   'stock_corr': stock_corr,
                   'portfolio_corr': portfolio_corr}
        else:
            return portfolio_volatility, stock_volatility, stock_prediction, available_notional, stock_notional, stock_corr, portfolio_corr

    def _features_to_state(self, portfolio_volatility, stock_volatility, stock_prediction,
                         available_notional, stock_notional, stock_corr, portfolio_corr):
        return np.array([portfolio_volatility * 100, stock_volatility * 100,
                         stock_prediction * 1e4, available_notional, stock_notional * 10, stock_corr * 10, portfolio_corr * 10])

    def _update_portfolio_volatility(self, allocation, portfolio_volatility, stock_volatility, initial_stock_notional, stock_corr, portfolio_corr):
        #Compute change in variance
        f1 = allocation * 2 * stock_corr * stock_volatility
        f2 = portfolio_corr * portfolio_volatility - initial_stock_notional * stock_corr * stock_volatility
        f3 = allocation **2 + 2*allocation*initial_stock_notional
        f4 = stock_volatility**2
        variance_change = f1*f2 + f3*f4
        #compute update in correlation
        initial_portfolio_volatility = portfolio_volatility
        portfolio_volatility = (initial_portfolio_volatility**2 + variance_change)**0.5
        p1 = portfolio_corr * initial_portfolio_volatility/portfolio_volatility
        p2 = stock_corr * stock_volatility * allocation / portfolio_volatility
        portfolio_corr = p1 + p2
        if math.isnan(portfolio_volatility):
            return 0, 0
        else:
            return portfolio_volatility, portfolio_corr

    def _start_saving_time(self, process_name):
        self._start = time.time()
        self._tracking_time = True
        self._tracking_process_name = process_name
    
    def _end_saving_time(self, process_name):
        if not self._tracking_time:
            raise Exception('Called _end_saving_time without being tracking time')
        self._tracking_time = False
        if process_name != self._tracking_process_name:
            raise Exception('Called _end_saving_time with a different process name')
        if not hasattr(self, '_dict_process_times'):
            self._dict_process_times = {}
        self._tracking_process_name = None
        total_time = time.time() - self._start
        if process_name in self._dict_process_times.keys():
            self._dict_process_times[process_name] += total_time
        else:
            self._dict_process_times[process_name] = total_time

    def _df_to_dict(self, df, key_name, complete = False):
        if complete:
            return {key: data.iloc[0, :].to_dict() for key, data in df.groupby(key_name)}
        else:
            return {key: data for key, data in df.groupby(key_name)}

class StockWithVolCorrVec(gym.Env, AuxiliarEnvMethods):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}


    def __init__(self, df, verbose = False, save_hidden_state_memory = True):
        #Cte parameters
        self.df = df.copy()
        self._vectorize(df)
        self.n_steps = len(self.df)
        self.verbose = verbose
        #Define state and action spaces
        self.action_space = spaces.Box(low = -1, high = 1,shape = (1,)) 
        self.observation_space = spaces.Box(low=-30, high=30, shape = (7,))
        #State variables
        self.reward = 0
        self.rewards_memory = []
        self.terminal = False  
        self.rl_timestamp = 0
        self.day = 0
        #Set state
        self.update_state(action = None)           
        self._seed(0)
        #Debug
        self.save_hidden_state_memory = save_hidden_state_memory
        self._set_debug_lists()
    
    def _vectorize(self, df):
        self.v_asset_identifier = df['asset_identifier'].to_list()
        self.v_date             = df['date'].to_list()
        self.v_volatility       = df['volatility'].to_list()
        self.v_rho              = df['rho'].to_list()
        self.v_yhat             = df['yhat'].to_list()
        self.v_target_return    = df['target_return'].to_list()

        d_date = self._df_to_dict(df, 'date')
        d_date_stock = {date: self._df_to_dict(data, 'asset_identifier', complete = True) for date, data in d_date.items()}
        self.d_date_stock = d_date_stock

    def step(self, action):
        action = action# *0.02
        self.state_memory.append(self._state_to_features(True))
        self.hidden_state_before_action_memory.append(deepcopy(self.hidden_state))
        self.action_memory.append(action.item())
        self.terminal = self.rl_timestamp == self.n_steps - 1
        if not self.terminal:
            self.reward = self.get_reward(action)
            self.rewards_memory.append(self.reward)
            #load next state
            self.update_state(action)
            self.hidden_state_after_action_memory.append(deepcopy(self.hidden_state))
        else:
            if self.verbose:
                print(f"Episode rewards: {sum(self.rewards_memory)}")
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.rl_timestamp = 0
        self.day = 0
        self.terminal = False 
        self.rewards_memory = []
        self._set_debug_lists()
        #initiate state
        self.update_state(action = None)
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self, action):
        portfolio_volatility, stock_volatility, _, available_notional, initial_stock_notional, stock_corr, portfolio_corr = self._state_to_features()
        #Transform the action into an allocation
        allocation = action.item()
        allocation, available_notional, stock_notional = self._bound_allocation(allocation, available_notional, initial_stock_notional)
        #Get the pnl reward
        stock_target_return = self.v_target_return[self.rl_timestamp]
        pnl_reward          = stock_notional * stock_target_return * 100 #Multiply by 100 to normalize
        #Get the vol reward
        # print(portfolio_volatility, initial_stock_notional, stock_volatility)
        portfolio_volatility, _ = self._update_portfolio_volatility(allocation, portfolio_volatility, stock_volatility, initial_stock_notional, stock_corr, portfolio_corr)
        # print(portfolio_volatility*100)
        vol_reward           = -0.1 * (portfolio_volatility > 0.5e-2)  #Just guessing a volatility threshold
        # vol_reward           = -1 * self._relu(portfolio_volatility, 0.5e-2)  #Just guessing a volatility threshold
        # print(pnl_reward, vol_reward)
        reward               = pnl_reward# + vol_reward
        return reward
    
    def update_state(self, action):
        #If action is None just initializate a position
        if action == None:
            portfolio_volatility = 0 #No positions
            portfolio_corr = 0
            available_notional = 1
            stock_notional = 0
            #Set hidden state
            self._set_hidden_state()
        else:
            #Get state features to further update
            portfolio_volatility, stock_volatility, _, available_notional, initial_stock_notional, stock_corr, portfolio_corr = self._state_to_features()
            #Transform the action into an allocation (carefully bound the allocation accordingly with the available notional)
            allocation = action.item()
            allocation, available_notional, stock_notional = self._bound_allocation(allocation, available_notional, initial_stock_notional)
            #Update the allocated notional in the hidden state
            stock = self.v_asset_identifier[self.rl_timestamp]
            self._update_hidden_state_notional(stock, stock_notional, action.item())
            #Update the portfolio volatility
            portfolio_volatility, portfolio_corr = self._update_portfolio_volatility(allocation, portfolio_volatility, stock_volatility, initial_stock_notional, stock_corr, portfolio_corr)
            #Make a RL step
            self.rl_timestamp += 1
            #If it is a new day, update the portfolio variables
            day = self.v_date[self.rl_timestamp]
            if day != self.day:
                # print(f'NEW DAY: {day}')
                if self.save_hidden_state_memory:
                    self.hidden_state_memory[self.day] = deepcopy(self.hidden_state)
                self.day = day
                portfolio_volatility, portfolio_corr, available_notional = self._update_hidden_state_new_day()
                # self._reorder_df_by_absolute_notional()
            elif self.rl_timestamp == len(self.df) - 1:
                if self.save_hidden_state_memory:
                    self.hidden_state_memory[self.day] = self.hidden_state.copy()
        #Get variables to the next stock
        new_stock            = self.v_asset_identifier[self.rl_timestamp]
        new_stock_volatility = self.v_volatility[self.rl_timestamp]
        new_stock_corr       = self.v_rho[self.rl_timestamp]
        new_stock_prediction = self.v_yhat[self.rl_timestamp]
        new_stock_notional   = self._get_current_notional(new_stock)
        # print({'portfolio_volatility': portfolio_volatility, 'new_stock_volatility': new_stock_volatility, 'new_stock_prediction': new_stock_prediction,
        #         'available_notional': available_notional, 'new_stock_notional': new_stock_notional})
        self.state = self._features_to_state(portfolio_volatility, new_stock_volatility, new_stock_prediction,
                                            available_notional, new_stock_notional, new_stock_corr, portfolio_corr)

class MarkEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MarkEpisodeCallback, self).__init__(verbose)
        self.epoch_ =  0

    def _on_step(self):
        if self.model.env.terminal:
            self.waiting_to_print = True
            self.epoch_ += 1
            print(f"End of epoch {self.epoch_}")
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, n_steps = None):
        super(TensorboardCallback, self).__init__(verbose)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        env = self.model.env.envs[0]
        if env.rl_timestamp == self.n_steps - 1:
            self.last_time_trigger = self.num_timesteps
            return self._on_event(env)
        return True
    
    def _on_event(self, env):
        self._record_scalar_variables(env)
        return True

    def _record_scalar_variables(self, env):
        #Rewards sumation
        sum_rewards = np.sum(env.rewards_memory)
        self.logger.record('principal/sum_rewards', sum_rewards)
        #Action mean over std
        action_mean_over_std = np.mean(env.action_memory)/np.std(env.action_memory)
        self.logger.record('env/action_mean_over_std', action_mean_over_std)
        #Action mean over std
        action_std = np.std(env.action_memory)
        self.logger.record('env/action_std', action_std)
        #Action correlation with target return
        corr = np.corrcoef(env.action_memory, env.v_target_return[:-1])[0, 1]
        self.logger.record('principal/action_correlation_with_target', corr)
        #Action correlation with all the state features
        state_variables = env.state_memory[0].keys()
        for variable in state_variables:
            folder = 'principal' if variable in ['stock_prediction', 'stock_notional'] else 'env'
            variable_observations = [state[variable] for state in env.state_memory]
            corr = np.corrcoef(env.action_memory, variable_observations)[0, 1]
            self.logger.record(f'{folder}/action_correlation_{variable}', corr)
        #Fraction of the time hands tied (without an available notional of at least 0.01)
        hands_tied = [state['available_notional'] < 0.01 for state in env.state_memory]
        hands_tied_fraction = np.mean(hands_tied)
        self.logger.record(f'principal/hands_tied_fraction', hands_tied_fraction)
        #Fraction of the actions higher than zero
        higher_than_zero_actions      = [action > 0 for action in env.action_memory]
        higher_than_zero_actions_mean = np.mean(higher_than_zero_actions)
        self.logger.record(f'principal/higher_than_zero_actions', higher_than_zero_actions_mean)
        #Fraction of the notionals higher than zero
        list_notionals = []
        for date, data in env.hidden_state_memory.items():
            list_notionals = list_notionals + [hidden_state[0] for hidden_state in data.values() if not math.isnan(hidden_state[1])]
        higher_than_zero_notionals = np.mean(np.array(list_notionals) > 0)
        self.logger.record(f'principal/higher_than_zero_notionals', higher_than_zero_notionals)
        #Fraction of allocations different from zero
        #Notional correlation with the target return
        notional = np.array([state['stock_notional'] for state in env.state_memory])
        corr = np.corrcoef(notional, env.v_target_return[:-1])[0, 1]
        self.logger.record(f'principal/notional_correlation_with_target', corr)
        #Rewards sum over equal_weight pnl
        sum_rewards = np.sum(env.rewards_memory)
        ew_pnl = env.df.groupby('date')['target_return'].mean().sum() * 100
        self.logger.record('principal/sum_rewards_over_ew_pnl', sum_rewards/ew_pnl)


def get_actions(env, model):
    episodes = 1
    list_actions = []
    for episode in range(episodes):
        done = False
        obs = env.reset()
        while not done:#not done:
            action, _ = model.predict(obs, deterministic = True)
            list_actions.append(action)
            obs, reward, done, info = env.step(action)
    return np.array(list_actions)

