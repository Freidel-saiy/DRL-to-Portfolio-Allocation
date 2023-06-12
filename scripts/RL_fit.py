from lib.RL_portfolio.pipeline import Pipeline

n_trials = int(input("\nEnter number of trials: "))
path_data   = str(input("Pass the folder with the data to be processed: "))
logdir = '../log'
### PATH PARAMETERS ################################################
pipe = Pipeline()
df_valid, df_test = pipe.read_data(path_data)
### REINFORCEMENT LEARNING FIT ################################################
pipe.fit_rl(df_valid, n_trials, logdir)



