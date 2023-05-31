import warnings

warnings.filterwarnings('ignore')

import sys
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import plotly

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from FinanceRobot_Backtest_lib import Dataset_Generator, Finance_Environment_V2, data_normalization
from FinanceRobot_Backtest_lib import Backtesting_vector, Backtesting_event
from FinanceRobot_DDQNPPOModel_lib import series_decomp, Decompose_FF_Linear, FinRobotAgentDQN

import numpy as np
import pandas as pd
from copy import deepcopy
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']    # for chinese text on plt
# # for chinese text negative symbol '-' on plt
# plt.rcParams['axes.unicode_minus'] = False


from BTCCrawl_To_DataFrame_Class import BTC_data_acquire as BTC_DataAcquire
from BTCCrawl_To_DataFrame_Class import get_api_key

if __name__ == '__main__':
    # 调用BTC爬取部分
    sys.path.append("e:/Python_WorkSpace/量化交易/")  # 增加指定的绝对路径,进入系统路径,从而便于该目录下的库调用
    Folder_base = "e:/Python_WorkSpace/量化交易/data/"
    config_file_path = "e:/Python_WorkSpace/量化交易/BTCCrawl_To_DataFrame_Class_config.ini"
    # URL = "https://api.coincap.io/v2/candles?exchange=binance&interval=h12&baseId=bitcoin&quoteId=tether"
    URL = 'https://data.binance.com'
    StartDate = "2023-1-20"
    EndDate = "2023-06-01"
    BTC_json = "BTC_h12.json"
    BinanceBTC_json = "BinanceBTC_h12.json"

    api_key, api_secret = get_api_key(config_file_path)

    BTC_data = BTC_DataAcquire(URL, StartDate, EndDate, Folder_base, BTC_json,
                               binance_api_key=api_key, binance_api_secret=api_secret)
    data = BTC_data.MarketFactor_ClosePriceFeatures(by_BinanceAPI=True,
                                                    FromWeb=False, close_colName='close', lags=0, window=20, horizon=5,
                                                    interval='12h', MarketFactor=True, weekdays=7)

    batch_size = 32
    data_normalized = data_normalization(data, 365, )
    dataset = Dataset_Generator(data_normalized)
    env = Finance_Environment_V2(dataset, action_n=3, min_performance=0., min_accuracy=0.1)  # 允许做空,允许大亏,使得更多的训练数据出现
    init_state, init_non_state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)  # state:(1,lags,obs_n)

    lags, obs_n = env.observation_space.shape
    action_n = env.action_space.n
    Q = Decompose_FF_Linear(seq_len=lags, in_features=obs_n, out_features=action_n, )
    Q_target = deepcopy(Q)
    # actions = model(state) # (N,1, action_n)
    FinR_Agent = FinRobotAgentDQN(Q, Q_target, gamma=0.98, learning_rate=5e-4, learn_env=env, fit_batch_size=64, )
    # 训练过程:
    # FinR_Agent.learn(episodes=100)
    # print(f"{'-' * 40}finished{'-' * 40}")
    # 生成env_backtest_data:
    env_backtest_data = env.dataset2data() (N,3)
    # print(env_backtest_data)



