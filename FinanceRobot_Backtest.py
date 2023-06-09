import warnings

warnings.filterwarnings('ignore')

import sys
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly

import tensorflow as tf

from FinanceRobot_Backtest_lib import Dataset_Generator, Finance_Environment_V2, data_normalization
from FinanceRobot_Backtest_lib import BacktestingVectorV2, BacktestingEventV2
from FinanceRobot_DDQNPPOModel_lib import series_decomp, Decompose_FF_Linear, FinRobotAgentDQN,FinRobotAgentDDQN

# import numpy as np
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
    EndDate = "2023-06-10"
    BTC_json = "BTC_h12.json"
    BinanceBTC_json = "BinanceBTC_h12.json"

    api_key, api_secret = get_api_key(config_file_path)

    BTC_data = BTC_DataAcquire(URL, StartDate, EndDate, Folder_base, BTC_json,
                               binance_api_key=api_key, binance_api_secret=api_secret)
    data = BTC_data.MarketFactor_ClosePriceFeatures(by_BinanceAPI=True,
                                                    FromWeb=False, close_colName='close', lags=0, window=20, horizon=10,
                                                    interval='12h', MarketFactor=True, weekdays=7)

    batch_size = 32
    data_normalized = data_normalization(data, 365, )
    dataset = Dataset_Generator(data_normalized, lags=7, shuffle=False)
    env = Finance_Environment_V2(dataset, action_n=3, min_performance=0., min_accuracy=0.1)  # 允许做空,允许大亏,使得更多的训练数据出现
    init_state, init_non_state = env.reset()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)  # state:(1,lags,obs_n)

    lags, obs_n = env.observation_space.shape
    action_n = env.action_space.n
    Q = Decompose_FF_Linear(seq_len=lags, in_features=obs_n, out_features=action_n, )
    Q_target = deepcopy(Q)
    # actions = model(state) # (N,1, action_n)
    FinR_Agent_DQN = FinRobotAgentDQN(Q, Q_target, gamma=0.50, learning_rate=5e-4, learn_env=env, memory_size=2000,
                                  replay_batch_size=1000, fit_batch_size=batch_size, )
    FinR_Agent_DDQN = FinRobotAgentDDQN(Q, Q_target, gamma=0.50, learning_rate=5e-4, learn_env=env, memory_size=2000,
                                  replay_batch_size=1000, fit_batch_size=batch_size, )

    today_date = pd.Timestamp.today().strftime('%y%m%d')

    DDQN_flag = True
    if DDQN_flag: # DDQN
        saved_path_prefix = 'saved_model/BTC_DDQN_'
        saved_path = saved_path_prefix + 'gamma05_lag7_' + today_date + ".h5"
        # DDQN 训练过程:
        FinR_Agent_DDQN.learn(episodes=70)
        # 最后训练模型h5格式存盘
        FinR_Agent_DDQN.Q.save_weights(saved_path,save_format='h5',overwrite=False)
    else: # DQN
        saved_path_prefix = 'saved_model/BTC_DQN_'
        saved_path = saved_path_prefix + 'gamma05_lag7_' + today_date + ".h5"
        # DQN 训练过程:
        FinR_Agent_DQN.learn(episodes=70)
        print(f"{'-' * 40}finished{'-' * 40}")
        # 最后训练模型h5格式存盘
        FinR_Agent_DDQN.Q.save_weights(saved_path,save_format='h5',overwrite=False)

    # 调出预训练模型:
    # ckpt = tf.train.Checkpoint(model=FinR_Agent.Q)
    # saved_path = saved_path_prefix + '230608-33'
    # ckpt.restore(saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,否则模型只是restore不成功,程序并不退出,浪费数天时间.

    # vector backtest
    # env_backtest_data  = BacktestingVectorV2(Q,env,)

    # Event Based Backtesting
    if DDQN_flag: # DDQN
        BacktestEvent = BacktestingEventV2(env, FinR_Agent_DDQN.Q, initial_amount=1000, percent_commission=0.001,
                                       fixed_commission=0., verbose=True, MinUnit_1Position=-8, )
    else: # DQN
        BacktestEvent = BacktestingEventV2(env, FinR_Agent_DQN.Q, initial_amount=1000, percent_commission=0.001,
                                           fixed_commission=0., verbose=True, MinUnit_1Position=-8, )
    BacktestEvent.backtest_strategy_WO_RM()
    # print(BacktestEvent.net_wealths)

    # plot 绘图:

    # net_wealth
    trace5 = go.Scatter(  #
        x=BacktestEvent.net_wealths.index,
        y=BacktestEvent.net_wealths['net_wealth'],
        mode="markers",  # mode模式
        name="净资产收益",
        showlegend=False,
        xhoverformat="%y/%m/%d_%H:00",
        yhoverformat=".2f",
        # hovertemplate='日期:%{x},价格: %{y:$.0f}',
    )

    # 收盘价:
    trace6 = go.Scatter(  #
        x=BacktestEvent.net_wealths.index,
        y=BacktestEvent.net_wealths['price'],  # 收盘价
        mode="markers",  # mode模式
        name="收盘价",
        showlegend=False,
        xhoverformat="%y/%m/%d_%H:00",
        yhoverformat="$,.0f",
        marker=dict(color=BacktestEvent.net_wealths['action'], line_width=0.5,
                    colorscale=['red', 'yellow', 'green'], showscale=True),
        # hovertemplate='日期:%{x},价格: %{y:$.0f}',
    )

    # units:
    trace7 = go.Scatter(  #
        x=BacktestEvent.net_wealths.index,
        y=BacktestEvent.net_wealths['units'],  # 股/币数
        mode="markers",  # mode模式
        name="股/币数",
        showlegend=False,
        xhoverformat="%y/%m/%d_%H:00",
        yhoverformat="B,.5f",
        marker=dict(color=BacktestEvent.net_wealths['action'], line_width=0.5,
                    colorscale=['red', 'yellow', 'green'], showscale=True),
        # hovertemplate='日期:%{x},价格: %{y:$.0f}',
    )

    # horizon_return_after_ptc:
    trace8 = go.Scatter(  #
        x=BacktestEvent.net_wealths.index,
        y=BacktestEvent.net_wealths['horizon_return_after_ptc'],  # (horizon_price-price(1-trade_commision))/price
        mode="markers",  # mode模式
        name="horizon涨跌幅",
        showlegend=False,
        xhoverformat="%y/%m/%d_%H:00",
        yhoverformat="R,.4f",
        marker=dict(color=BacktestEvent.net_wealths['action'], line_width=0.5,
                    colorscale=['red', 'yellow', 'green'], showscale=True),
        # hovertemplate='日期:%{x},价格: %{y:$.0f}',
    )
    layout = dict(
        title=dict(text='事件型回测:', font=dict(
            color='rgb(0,125,125)', family='SimHei', size=20)),
        margin=dict(l=50, b=10, t=50, r=15, pad=0),
        # xaxis=dict(title="交易日期", tickangle=-30, tickformat='%y/%m/%d_%H:'),  # 设置坐标轴的标签
    )

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[
        '模型策略净资产收益', '收盘价及策略action', '头寸及策略action', '除交易费后的horizon预期收益率'], )
    fig.add_traces(data=[trace5], rows=1, cols=1, )
    fig.add_traces(data=[trace6], rows=2, cols=1, )
    fig.add_traces(data=[trace7], rows=3, cols=1, )
    fig.add_traces(data=[trace8], rows=4, cols=1, )
    fig.update_xaxes(tickangle=-30, tickformat='%y/%m/%d_%H:', row=1, col=1, )
    fig.update_yaxes(title="净资产", tickformat=',.0f', row=1, col=1)
    fig.update_yaxes(title="收盘价及策略action", tickformat='', row=2, col=1)
    fig.update_yaxes(title="股/币数", tickformat='', row=3, col=1)
    fig.update_yaxes(title="horizon涨跌幅", tickformat='', row=4, col=1)
    fig.update_layout(layout)
    # fig.show()

    # 获取回测数据最后一个交易日日期.
    last_date = BacktestEvent.net_wealths.index[-1]
    last_date = last_date.strftime('%y%m%dH%H')
    saved_path = '{}gamma05_lag7_{}.html'.format(saved_path_prefix, last_date)

    fig.write_html(saved_path, include_plotlyjs=True, auto_open=False)
