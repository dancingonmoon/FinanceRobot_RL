## Reinforcement Learning Algorithms Applications for BTC Trading

the project is to apply DQN, DDQN and PPO upon BTC trading. it accomplishes below tasks:
+ download OHLC data from encryption exchange house;
+ generate features data via OHLC data, which include:
    + volume
    + RSI
    + Log_close_weeklag
    + Log_high_low
    + Log_open_weeklag
    + open_pre_close
    + high_pre_close
    + low_pre_close
    + num_trades
    + bid_volume
    + horizon_price
    + close
+ data normalization with previous 365 lookback
+ set up environment, with the class Name: Finance_Environment_V2
  + action_space: long and short are all applied during the course of training, while in the course of backtest, only short is not advised. 
    + buy = 2
    + hold = 1
    + sell = 0
  + reward function: 
    + reward1: positive return makes reward1 = 1 , while negative return makes reward1 = -1; reward1 = 0, when action is on hold;
    + reward2: np.log(margin_ratio) * (action - 1)
    + reward = reward1 + reward2 * 5
+ Training either by DQN , or DDQN , or PPO, and save the trained_model
+ Backtesting:
  + vector Backtesting;
  + event based Backtesting: short is not recommended;
  + to produce plotly chart 