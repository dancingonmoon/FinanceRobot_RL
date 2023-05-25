import warnings
warnings.filterwarnings('ignore')

import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

import tensorflow as tf

# from sklearn.preprocessing import MinMaxScaler
from FinanceRobot_Backtest_lib import Dataset_Generator, Finance_Environment, FQLAgent,build_model, data_normalization
from FinanceRobot_Backtest_lib import Backtesting_vector, Backtesting_event
from FinanceRobot_Backtest_lib import series_decomp,Decompose_FF_Linear

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']    # for chinese text on plt
# # for chinese text negative symbol '-' on plt
# plt.rcParams['axes.unicode_minus'] = False

# 调用BTC爬取部分
sys.path.append("L:/Python_WorkSpace/量化交易/")  # 增加指定的绝对路径,进入系统路径,从而便于该目录下的库调用

from BTCCrawl_To_DataFrame_Class import BTC_data_acquire as BTC_DataAcquire
from BTCCrawl_To_DataFrame_Class import get_api_key
Folder_base = "L:/Python_WorkSpace/量化交易/data/"
config_file_path = "L:/Python_WorkSpace/量化交易/BTCCrawl_To_DataFrame_Class_config.ini"
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
                                                FromWeb=False, close_colName='close', lags=0, window=20, interval='12h',MarketFactor=True, weekdays=7)

# print(data.columns)

data_normalized = data_normalization(data, 356,)
print(data_normalized.describe()) # features_num=18