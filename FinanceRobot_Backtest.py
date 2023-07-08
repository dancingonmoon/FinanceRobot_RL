import warnings

warnings.filterwarnings('ignore')

import sys
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import plotly

import tensorflow as tf

# tf.config.set_visible_devices([], 'GPU') # 让GPU不可见,仅仅使用CPU

# 获取所有可见的物理设备
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    if device.device_type == 'GPU':
        tf.config.set_visible_devices(device, 'GPU')
        # 将显存设置为动态增长模式 (可以避免多进程中,单个进程GPU内存全部分配而崩塌)
        tf.config.experimental.set_memory_growth(device, True)

from FinanceRobot_Backtest_lib import Dataset_Generator, ndarray_Generator, TupleIterator, Finance_Environment_V2, \
    data_normalization
from FinanceRobot_Backtest_lib import BacktestingVectorV2, BacktestingEventV2
from FinanceRobot_DDQNModel_lib import Decompose_FF_Linear, FinRobotAgentDQN, FinRobotAgentDDQN
from FinanceRobot_PPOModel_lib import Worker, ActorModel, CriticModel, PPO2

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
    sys.path.append("l:/Python_WorkSpace/量化交易/")  # 增加指定的绝对路径,进入系统路径,从而便于该目录下的库调用
    Folder_base = "l:/Python_WorkSpace/量化交易/data/"
    config_file_path = "l:/Python_WorkSpace/量化交易/BTCCrawl_To_DataFrame_Class_config.ini"
    # URL = "https://api.coincap.io/v2/candles?exchange=binance&interval=h12&baseId=bitcoin&quoteId=tether"
    URL = 'https://data.binance.com'
    StartDate = "2023-1-20"
    EndDate = "2023-09-10"
    BTC_json = "BTC_h12.json"
    BinanceBTC_json = "BinanceBTC_h12.json"

    api_key, api_secret = get_api_key(config_file_path)

    BTC_data = BTC_DataAcquire(URL, StartDate, EndDate, Folder_base, BTC_json,
                               binance_api_key=api_key, binance_api_secret=api_secret)
    horizon = 2 # best for DDQN 14; PPO 2
    lookback = 225 #best for DDQN 225; PPO 225;
    MarketFactor = False #best for DDQN True; PPO False

    data = BTC_data.MarketFactor_ClosePriceFeatures(by_BinanceAPI=True,
                                                    FromWeb=False, close_colName='close', lags=0, window=20, horizon=horizon,
                                                    interval='12h', MarketFactor=MarketFactor, weekdays=7)
    if MarketFactor:
        normalize_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        normalize_columns = [0, 1, 2, 3, 4, 5]
    data_normalized = data_normalization(data, lookback, normalize_columns=normalize_columns)
    # features = [
    #             "log_return",
    #             "Roll_price_sma",
    #             "Roll_price_min",
    #             "Roll_price_max",
    #             "Roll_return_mom",
    #             "Roll_return_std",
    #             "horizon_price",
    #             close_colName,
    #         ]
    # split 训练数据,验证数据:
    split = np.argwhere(data_normalized.index == pd.Timestamp('2023-06-01', tz='UTC'))[0, 0]

    #########Arguments Optimization#############
    Test_flag = True
    train_test_text_add = 'test' if Test_flag else 'train'
    Train_with_Pretrained_model = False

    DQN_DDQN_PPO = 'PPO' # 或者"DQN", "PPO"
    lags = 5 # best for DDQN 7; PPO 5
    action_n = 3
    gamma = 0.5 # best for DDQN 0.98; PPO 0.5
    memory_size = 512
    replay_batch_size = 256
    batch_size = 16 # best for DDQN 16; PPO 16
    DQN_episode = 80
    DDQN_episode = 80

    DQN_saved_model_filename = "230610-51"
    DDQN_saved_model_filename = "230701-39"

    # PPO部分
    n_worker = 8
    n_step = 3
    mini_batch_size = int(n_worker*n_step/4)  # int(n_worker * n_step / 4)
    gae_lambda = 0.9
    gradient_clip_norm = .2
    epochs = 3
    actor_lr= 1e-3
    critic_lr= 5e-3
    updates = 10000
    today_date = pd.Timestamp.today().strftime('%y%m%d')

    PPO_saved_model_filename = "230707-7"
    ####################

    if DQN_DDQN_PPO == "DQN" or DQN_DDQN_PPO == "DDQN":
        # 生成dataset, env, 建模
        dataset_train = Dataset_Generator(data_normalized[:split], lags=lags,
                                          data_columns_state=normalize_columns)
        dataset_test = Dataset_Generator(data_normalized[split:], lags=lags,
                                         data_columns_state=normalize_columns)
        # 训练environment, 测试environment:
        env = Finance_Environment_V2(dataset_train, dataset_type='tensorflow_Dataset', action_n=action_n, min_performance=0.,
                                     min_accuracy=0.0)  # 允许做空,允许大亏,使得更多的训练数据出现
        env_test = Finance_Environment_V2(dataset_test, dataset_type='tensorflow_Dataset', action_n=action_n, min_performance=0.,
                                          min_accuracy=0.0)  # 允许做空,允许大亏,使得更多的训练数据出现

        # 生成模型,以及FinR_Agent:
        lags, obs_n = env.observation_space.shape

        Q = Decompose_FF_Linear(seq_len=lags, in_features=obs_n, out_features=action_n, )
        Q_target = deepcopy(Q)
        # actions = model(state) # (N,1, action_n)
        FinR_Agent_DQN = FinRobotAgentDQN(Q, Q_target, gamma=gamma, learning_rate=5e-4, learn_env=env,
                                          memory_size=memory_size,
                                          replay_batch_size=replay_batch_size, fit_batch_size=batch_size, )
        FinR_Agent_DDQN = FinRobotAgentDDQN(Q, Q_target, gamma=gamma, learning_rate=5e-4, learn_env=env,
                                            memory_size=memory_size,
                                            replay_batch_size=replay_batch_size, fit_batch_size=batch_size, )
        # 生成FinR_Agent 训练,存盘,调出训练模型
        if DQN_DDQN_PPO == 'DDQN':  # DDQN
            saved_path_prefix = 'saved_model/BTC_DDQN_'
            saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
            if not Test_flag:
                # DDQN 训练过程:
                FinR_Agent_DDQN.learn(episodes=DDQN_episode)
                print(f"{'-' * 40}finished{'-' * 40}")
                # 最后训练模型h5格式存盘
                FinR_Agent_DDQN.Q.save_weights(saved_path, save_format='h5', overwrite=False)
                input_env = env
                model = Q
                action_strategy_mode = 'argmax'

            elif Test_flag:
                # 调出预训练模型, event based backtesting:
                ckpt = tf.train.Checkpoint(model=Q)
                saved_path = saved_path_prefix + DDQN_saved_model_filename
                ckpt.restore(
                    saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,否则模型只是restore不成功,程序并不退出,浪费数天时间.
                input_env = env_test
                model = Q
                action_strategy_mode = 'argmax'


        elif DQN_DDQN_PPO == 'DQN':  # DQN
            saved_path_prefix = 'saved_model/BTC_DQN_'
            saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
            if not Test_flag:
                # DDQN 训练过程:
                FinR_Agent_DQN.learn(episodes=DQN_episode)
                print(f"{'-' * 40}finished{'-' * 40}")
                # 最后训练模型h5格式存盘
                FinR_Agent_DQN.Q.save_weights(saved_path, save_format='h5', overwrite=False)
                input_env = env
                model = Q
                action_strategy_mode = 'argmax'

            elif Test_flag:
                # 调出预训练模型, event based backtesting:
                ckpt = tf.train.Checkpoint(model=Q)
                saved_path = saved_path_prefix + DQN_saved_model_filename
                ckpt.restore(
                    saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,否则模型只是restore不成功,程序并不退出,浪费数天时间.
                input_env = env_test
                model = Q
                action_strategy_mode = 'argmax'
            # DQN 训练过程:
            FinR_Agent_DQN.learn(episodes=70)
            print(f"{'-' * 40}finished{'-' * 40}")


    elif DQN_DDQN_PPO == 'PPO':  # PPO
        dataset_train = ndarray_Generator(data_normalized[:split], lags=lags,
                                          data_columns_state=normalize_columns)
        dataset_test = ndarray_Generator(data_normalized[split:], lags=lags,
                                         data_columns_state=normalize_columns)
        iter_dataset_train = TupleIterator(dataset_train)
        iter_dataset_test = TupleIterator(dataset_test)
        # 训练environment, 测试environment:
        env = Finance_Environment_V2(iter_dataset_train, dataset_type='ndarray_iterator', action_n=action_n,
                                     min_performance=0.,
                                     min_accuracy=0.1)  # 允许做空,允许大亏,使得更多的训练数据出现
        env_test = Finance_Environment_V2(iter_dataset_test, dataset_type='ndarray_iterator', action_n=action_n,
                                          min_performance=0.,
                                          min_accuracy=0.1)  # 允许做空,允许大亏,使得更多的训练数据出现
        _, obs_n = env.observation_space.shape

        # PPO 建模:
        saved_path_prefix = 'saved_model/BTC_PPO_'

        Actor = ActorModel(seq_len=lags, in_features=obs_n, out_features=action_n)
        Critic = CriticModel(seq_len=lags, in_features=obs_n, out_features=action_n)

        if not Test_flag:
            # 训练用 多进程 建模
            workers = []
            for i in range(n_worker):
                worker = Worker(dataset=iter_dataset_train, dataset_type='ndarray_iterator', action_dim=action_n)
                workers.append(worker)
            FinR_Agent_PPO = PPO2(workers, Actor, Critic, action_n, lags, obs_n, actor_lr=actor_lr, critic_lr=critic_lr,
                                  gae_lambda=gae_lambda,gamma=gamma,
                                  c1=1., gradient_clip_norm=gradient_clip_norm, n_worker=n_worker, n_step=n_step, epochs=epochs,
                                  mini_batch_size=mini_batch_size)
            saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
            if Train_with_Pretrained_model: # 调用预训练模型
                ckpt = tf.train.Checkpoint(actormodel=Actor, criticmodel=Critic)
                saved_path = saved_path_prefix + PPO_saved_model_filename
                ckpt.restore(
                    saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,否则模型只是restore不成功,程序并不退出,浪费数天时间.
            # PPO 训练过程:
            FinR_Agent_PPO.nworker_nstep_training_loop(updates=updates)
            FinR_Agent_PPO.close_process()
            print(f"{'-' * 40}finished{'-' * 40}")
            input_env = env
            model = Actor
            action_strategy_mode = 'tfp.distribution'

        elif Test_flag:
            # 调出预训练模型, event based backtesting:
            ckpt = tf.train.Checkpoint(actormodel=Actor, criticmodel=Critic)
            saved_path = saved_path_prefix + PPO_saved_model_filename
            ckpt.restore(
                saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,否则模型只是restore不成功,程序并不退出,浪费数天时间.
            input_env = env_test
            model = Actor
            action_strategy_mode = 'tfp.distribution'

    BacktestEvent = BacktestingEventV2(input_env, model, initial_amount=1000, percent_commission=0.001,
                                       fixed_commission=0., verbose=True, MinUnit_1Position=-8, )
    BacktestEvent.backtest_strategy_WO_RM(action_strategy_mode=action_strategy_mode)

    # init_state, init_non_state = env.reset()
    # action = env.action_space.sample()
    # state, reward, done, info = env.step(action)  # state:(1,lags,obs_n)

    # vector backtest
    # env_backtest_data  = BacktestingVectorV2(Q,env,)

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

    saved_path = '{}gamma0{}_lag{}_{}_{}.html'.format(saved_path_prefix, str(int(gamma * 100)), lags,
                                                      train_test_text_add, last_date)
    # 将BacktestEvent.net_wealths (pandas) 存盘:
    # BacktestEvent.net_wealths.to_json(f'{saved_path_prefix}_netwealths_{train_test_text_add}_{last_date}.json')

    fig.write_html(saved_path, include_plotlyjs=True, auto_open=False)
