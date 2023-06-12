import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
plt.style.use('dark_background')
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import os
from IPython.display import clear_output

import glob
from lib.RL_portfolio.RL_trading_envs_vec import StockWithVolCorrVec, TensorboardCallback
from lib.RL_portfolio.utilities  import Portfolio, Metrics


class Pipeline():
    def __init__(self):
        pass
    ### Pre RL part of the pipeline
    def make_predictions(self, df, data_parameters, date_parameters, model_parameters):
        #Prepare data
        print('Preparing data')
        df, features          = format_df_to_lstm(df, **data_parameters)
        data_train, data_test = split_train_test(df, features = features, **date_parameters)
        #Fit model
        print('Fitting model')
        X_train, y_train      = get_X_y_lstm(data_train, features)
        X_test, y_test        = get_X_y_lstm(data_test, features)
        model, _              = fit_model_lstm(X_train, y_train, **model_parameters)
        #Predict
        print("Making predictions")
        yhat_train = model.predict(X_train)
        train_performance = np.corrcoef(y_train, yhat_train.reshape(-1))[0, 1]
        print(f"train performance (correlation): {train_performance}")
        yhat_test = model.predict(X_test)
        test_performance = np.corrcoef(y_test, yhat_test.reshape(-1))[0, 1]
        print(f"test performance (correlation): {test_performance}")

        data_train.loc[:, 'yhat'] = yhat_train
        data_test.loc[:, 'yhat']  = yhat_test
        return data_train, data_test
    
    def compute_risk_variables(self, df, df_spy, risk_parameters):
        #Get rolling volatility
        print("Computing rolling volatility")
        vol_window = risk_parameters.get('vol_window')
        df         = compute_volatility(df, vol_window)
        #Getting rolling correlation with spy
        print("Computing rolling correlation with spy")
        corr_window = risk_parameters.get('corr_window')
        df          = compute_spy_corr(df, df_spy, corr_window)
        return df
    
    def apply_allocations(self, df, allocation_parameters):
        df = df.copy()
        df.loc[:, 'weight'] = 0
        df.loc[:, 'selection'] = 0

        stock_selection = allocation_parameters.get('stock_selection')
        if (stock_selection is None) or (stock_selection == 'all'):
            df.loc[:, 'selection'] == 1
        elif stock_selection == 'long':
            n_daily_trades = allocation_parameters.get('n_daily_trades')
            df.loc[:, '_rank_yhat'] = df.groupby('date')['yhat'].rank(pct = False, ascending = False)
            long_entries            = df['_rank_yhat'] < n_daily_trades
            df.loc[long_entries, 'selection'] = 1
            del df['_rank_yhat']
        elif stock_selection == 'long_short':
            n_daily_trades = allocation_parameters.get('n_daily_trades')
            df.loc[:, '_rank_yhat'] = df.groupby('date')['yhat'].rank(pct = False, ascending = False)
            long_entries            = df['_rank_yhat'] < n_daily_trades/2 + 1
            df.loc[:, '_rank_yhat'] = df.groupby('date')['yhat'].rank(pct = False, ascending = True)
            short_entries           = df['_rank_yhat'] < n_daily_trades/2 + 1
            df.loc[long_entries, 'selection']  = 1
            df.loc[short_entries, 'selection'] = -1
            del df['_rank_yhat']
        else:
            raise Exception("Invalid parameter stock selection")
        
        allocation_logic = allocation_parameters.get('allocation_logic')
        if allocation_logic == 'equal_weight':
            df.loc[:, 'weight'] = df.groupby('date')['selection'].transform(lambda x: x/x.abs().sum())
        elif allocation_logic == 'inverse_volatility':
            df.eval("weight = selection / volatility", inplace = True)
            df.loc[:, 'weight'] = df.groupby('date')['weight'].transform(lambda x: x/x.abs().sum())
        elif allocation_logic == 'min_variance':
            list_df = []
            for date, data in df.groupby('date'):
                data.query("selection.abs() == 1", engine = 'python', inplace = True)
                cov_matrix = Portfolio.get_cov_matrix(data)
                w          = Portfolio.min_variance(cov_matrix)
                data.loc[:, 'weight'] = w
                data.loc[:, 'weight']   = data.groupby('date')['weight'].transform(lambda x: x/x.abs().sum())
                list_df.append(data)
            df = pd.concat(list_df)
        
        target_vol = allocation_parameters.get('target_vol')
        if target_vol:
            list_df = []
            for date, data in df.groupby('date'):
                cov_matrix = Portfolio.get_cov_matrix(data)
                w          = data['weight'].values
                vol        = Metrics.compute_variance(w, cov_matrix)**0.5
                if vol > target_vol:
                    data.loc[:, 'weight'] = data['weight'] * target_vol/vol
                list_df.append(data)
            df = pd.concat(list_df)


        df.eval("pnl = weight * target", inplace = True)
        
        return df

    ### RL fit part of the pipeline
    def read_data(self, path_data):
        #Read inputs
        path_stocks = os.path.join(path_data, 'df_stocks_variables.pkl')
        while not (os.path.exists(path_stocks)):
            print("in the folder that you pass, there should be a df_stocks_variables.pkl")
            path_data   = str(input("Pass the folder with the data to be processed: "))
            path_stocks = os.path.join(path_data, 'df_stocks_variables.pkl')

        ### DATA TREATMENT ################################################
        print('Minor data manipulations...')
        df_valid, df_test = process_data(path_stocks, split_date = '2015-12-31')
        df_valid, df_test = clean_data(df_valid), clean_data(df_test)

        return df_valid, df_test
    
    def fit_rl(self, df_valid, n_trials, logdir):
        print("Fitting agent...")
        for i in range(n_trials):
            print(f"\nTrial {i + 1}")
            agent_parameters = {'seed': 3 + i, 
                                'gamma': 1,
                                'batch_size': int(2e4)}
            raw_env  = StockWithVolCorrVec
            model = self.fit_rl_ppo(df_valid, agent_parameters, raw_env, logdir = logdir)
    
    def fit_rl_ppo(self, df_valid, agent_parameters, raw_env,
               logdir = ""):
        #It will get the model label from the log_dir and use it to save new logs and to save a new model (with a new label)
        print('Creating env')
        env   = raw_env(df_valid, verbose = True)
        env   = Monitor(env)
        tensorboard_log = (os.path.join(logdir, 'tensorboard_logdir'))
        model = PPO('MlpPolicy', env, verbose=0, tensorboard_log = tensorboard_log, **agent_parameters)
        list_results = []
        tb_log_name = get_log_name(tensorboard_log)
        path_save_model = os.path.join(logdir, f'models_save/model_{tb_log_name}')
        create_folder(path_save_model)
        #save model callback (not using, saving by hand)
        eoe_checkpoint_callback = CheckpointCallback(save_freq=1, save_path=path_save_model, name_prefix='model_ep_')
        eoe_checkpoint_callback = EveryNTimesteps(n_steps=len(df_valid), callback=eoe_checkpoint_callback) #end of episode
        #stop training callback
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=0, min_evals=0, verbose=1)
        stop_train_callback = EvalCallback(env, eval_freq=100*len(df_valid), n_eval_episodes=1, callback_after_eval=stop_train_callback, verbose=1)
        #print epoch callback
        tensor_board_callback = TensorboardCallback(n_steps = len(df_valid))
        #train
        print("Training process start...\n")
        clear_output(wait=True)
        model.learn(total_timesteps=int(1e3 * len(df_valid)),
                    reset_num_timesteps = False,
                    callback=[stop_train_callback, tensor_board_callback],
                    tb_log_name = tb_log_name)
        model.save(os.path.join(logdir, f'models_save/model_by_hand_{tb_log_name}'))
        return model

    ### RL evaluation part of the pipeline
    def evolve_env(self, df_valid, model, raw_env):
        env = raw_env(df_valid, verbose = True, save_hidden_state_memory = True)
        done = False
        obs = env.reset()
        list_actions = []
        #Loop through env
        while not done:#not done:
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, done, _ = env.step(action)
            list_actions.append(action.item())
        #Extract information from hidden state memory
        list_hidden_state = []
        for day, hidden_state in env.hidden_state_memory.items():
            hidden_state = pd.DataFrame(hidden_state).T.reset_index()
            hidden_state.columns = ['asset_identifier', 'notional', 'vol', 'action', 'rho']
            hidden_state.loc[:, 'date'] = day
            list_hidden_state.append(hidden_state)
        hidden_state = pd.concat(list_hidden_state)
        return list_actions, hidden_state, env
    
    def evaluate_model(self, df_valid, model):
        #Run simulation on df_valid
        raw_env  = StockWithVolCorrVec
        _, hidden_state, env = self.evolve_env(df_valid, model, raw_env)
        # clear_output(wait = True)
        sum_rewards = np.sum(env.rewards_memory)
        print(sum_rewards)
        #Check correlation between actions and alocations with yhat
        df_merge = df_valid.merge(hidden_state, on = ['asset_identifier', 'date'])
        action_corr = df_merge[['action', 'yhat']].corr().iloc[0, 1]
        notional_corr = df_merge[['notional', 'yhat']].corr().iloc[0, 1]
        print("-- ** -- ** -- **")
        print("action correlation with yhat:   " + str(action_corr))
        print("notional correlation with yhat: " + str(notional_corr))
        print("-- ** -- ** -- **")
        #Plot action distribution
        plt.figure()
        plt.hist(df_merge['action'])
        plt.xlabel("action")
        plt.title("action distribution")
        #Plot pnl performance
        plt.figure(figsize = (14, 7))
        plt.subplot2grid(shape = (3, 1), loc = (0, 0))
        df_merge.loc[:, 'average_notional'] = df_merge['notional'].abs().mean()
        df_merge.eval("pnl = notional * target_return", inplace = True)
        df_merge.eval("pnl_average = average_notional * target_return", inplace = True)
        df_merge.groupby('date_backup')['pnl'].sum().cumsum().plot(label = 'agent_performance')
        df_merge.groupby('date_backup')['pnl_average'].sum().cumsum().plot()
        plt.legend()
        plt.grid()
        #Plot notional
        plt.subplot2grid(shape = (3, 1), loc = (1, 0))
        df_merge.groupby('date_backup')['notional'].apply(lambda x: x.abs().sum()).plot(label = 'notional gross')
        df_merge.groupby('date_backup')['notional'].apply(lambda x: x.sum()).plot(label = 'notional net')
        plt.legend()
        plt.grid()
        #Plot volatility
        df_merge.eval("variance = (notional * vol)**2", inplace = True)
        port_var = df_merge.groupby('date_backup')['variance'].sum()
        port_vol = port_var**0.5
        df_merge.eval("average_variance = (average_notional * vol)**2", inplace = True)
        average_var = df_merge.groupby('date_backup')['average_variance'].sum()
        average_vol = average_var**0.5
        plt.subplot2grid(shape = (3, 1), loc = (2, 0))
        plt.plot(port_vol, label = 'portfolio volatility')
        plt.plot(average_vol, label = 'equal weight volatility')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        return df_merge, env

### Methods to compute risk variables ##############################################################
def compute_spy_corr(df, df_spy, window):
    df.loc[:, 'ret']         = df.groupby("asset_identifier")['close'].pct_change()
    df_spy.loc[:, 'spy_ret'] = df_spy['close'].pct_change()
    df_spy                   = df_spy[['date', 'spy_ret']]
    df_merge                 = df.merge(df_spy, on = 'date', how = 'left')
    list_df = []
    for asset, data in df_merge.groupby('asset_identifier'):
        rolling_corr = data[['ret', 'spy_ret']].rolling(window).corr()
        rolling_corr = rolling_corr.reset_index(level = 1).query("level_1 == 'ret'")['spy_ret']
        data.loc[:, 'rho'] = rolling_corr
        data.loc[:, 'rho'] = data['rho'].shift(1)
        list_df.append(data)
    df = pd.concat(list_df).reset_index(drop = True)
    del df['ret']
    del df['spy_ret']
    return df

def compute_volatility(df, window):
    df.loc[:, 'ret'] = df.groupby("asset_identifier")['close'].pct_change()
    df.loc[:, 'volatility'] = df.groupby('asset_identifier')['ret'].apply(lambda x: x.rolling(window).std())
    df.loc[:, 'volatility'] = df.groupby('asset_identifier')['volatility'].shift(1)
    del df['ret']
    return df

#Methods to make predictions ########################################################################
def fit_model_lstm(X_train, y_train, lstm_parameters, optmizer_parameters, fit_parameters):
    model = Sequential()
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]), **lstm_parameters))
    model.add(Dense(1))
    model.compile(**optmizer_parameters)
    history = model.fit(X_train, y_train, **fit_parameters)
    return model, history

def get_X_y_lstm(data, features):
    X = data[features].values.reshape((-1, len(features), 1))
    y = data['target'].values
    return X, y

def split_train_test(df, train_start_date, gap_train_test, test_start_date, features = []):
    train_end_date = pd.to_datetime(test_start_date) - pd.to_timedelta(gap_train_test, unit = 'D')
    train_end_date = str(train_end_date.date())
    data_train = df.query("date > @train_start_date & date < @train_end_date").dropna(subset = features  + ['target'])
    data_test  = df.query("date > @test_start_date").dropna(subset = features  + ['target'])
    return data_train, data_test

def split_sequence(sequence, n_steps):
    list_subsequences = []
    empty_subsequence = [np.nan] * n_steps
    # Fill the initial subsequences with nan
    for i in range(min(n_steps, len(sequence))):
        list_subsequences.append(empty_subsequence)
    if len(sequence) <= n_steps:
        return list_subsequences
    # Fill the remaining subsequences
    for i in range(n_steps, len(sequence)):
        subsequence = sequence[i - n_steps:i]
        list_subsequences.append(subsequence)
    return list_subsequences

def format_df_to_lstm(df, n_steps):
    features = ['t_minus_{}'.format(n_steps - i) for i in range(n_steps)]
    df.loc[:, 'ret'] = df.groupby('asset_identifier')['close'].pct_change()
    list_df = []
    for asset, data in df.groupby('asset_identifier'):
        ret_shift_as_seq = split_sequence(data.loc[:, 'ret'].to_list(), n_steps)
        ret_shift_as_seq = pd.DataFrame(ret_shift_as_seq, index = data.index, columns = features)
        data = data.join(ret_shift_as_seq)
        list_df.append(data)

    df = pd.concat(list_df)
    df.loc[:, 'target'] = df.groupby('asset_identifier')['ret'].shift(-1)
    return df, features

### Other auxiliar methods
def clean_data(df_valid):#Warning - the data should come here already clean
    df_valid.query("rho > -1 & rho < 1 & volatility != 0", inplace = True)
    df_valid.sort_values(['date', 'asset_identifier'], inplace = True)
    df_valid.reset_index(inplace = True, drop = True)
    return df_valid
def process_data(path_data, split_date, start_date = '2004-01-01'):
    df = pd.read_pickle(path_data)

    # df.loc[:, 'volatility'] = df.groupby('asset_identifier')['t_minus_1'].apply(lambda x: x.rolling(35).std())

    df                           = df[['asset_identifier', 'date', 'rho', 'volatility', 'yhat', 'target']]
    df                           = df.rename(columns = {'target': 'target_return'})
    df.loc[:, 'realized_return'] = df.groupby("asset_identifier")['target_return'].shift(1)
    df                           = df.dropna()

    df.sort_values('date', inplace = True)
    df.loc[:,'date_backup'] = df['date'].copy()
    df.reset_index(drop = True, inplace = True)
    # df.loc[:, 'date']            = df['date'].factorize()[0]
    df.loc[:, 'asset_identifier'] = df['asset_identifier'].factorize()[0]

    df_test  = df.query("date > @split_date").reset_index(drop = True)
    df_test.loc[:, 'date'] = df_test['date'].factorize()[0]
    df_test.sort_values(['date', 'asset_identifier'], inplace = True)
    df_test.reset_index(drop = True, inplace = True)

    df_valid = df.query("date < @split_date & date > @start_date").reset_index(drop = True)
    df_valid.loc[:, 'date'] = df_valid['date'].factorize()[0]
    df_valid.sort_values(['date', 'asset_identifier'], inplace = True)
    df_valid.reset_index(drop = True, inplace = True)
    return df_valid, df_test


def get_log_name(logdir):
    folders = glob.glob(logdir + '/*')
    ppo_names = [folder.split('\\')[-1] for folder in folders]
    ppo_names = [name for name in ppo_names if 'PPO' in name]
    if len(ppo_names) == 0:
        return 'PPO_0'
    ppo_index = [int(name.split('_')[-2]) for name in ppo_names]
    ppo_index = max(ppo_index) + 1
    tb_log_name = 'PPO_' + str(ppo_index)
    return tb_log_name

def create_folder(path):
    if os.path.isdir(path):
        pass
    else:
        print('Creating folder: ' + path)
        os.makedirs(path)

def sweep_feature(model, env, features_base_values, sweep_dict):
    feature_to_sweep = list(sweep_dict.keys())[0]
    values_to_sweep = sweep_dict[feature_to_sweep]
    list_action = []
    for value in values_to_sweep:
        features = features_base_values.copy()
        features[feature_to_sweep] = value
        obs = env.features_to_state(**features)
        action, _ = model.predict(obs, deterministic = True)
        list_action.append(action.item())
    #Plot action vs feature swept
    plt.figure(figsize = (8, 3))
    plt.plot(values_to_sweep, list_action)
    plt.xlabel(feature_to_sweep)
    plt.ylabel("action")
    plt.grid()