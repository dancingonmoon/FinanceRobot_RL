# 使用Keras.Tuner;构建搜索空间;参数优化;
import warnings

warnings.filterwarnings('ignore')

import sys
import keras_tuner  # 由于该环境下protobuf原版本为3.17.0,运行出错,查stackoverlfow告知,升级至3.20.3,可运行,但pip安装出现一些不兼容
import tensorflow as tf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json
import glob

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

from BTCCrawl_To_DataFrame_Class import BTC_data_acquire as BTC_DataAcquire
from BTCCrawl_To_DataFrame_Class import get_api_key


def FinRobotSearchSpace(
        horizon=10,
        lookback=365,
        MarketFactor=False,
        DQN_DDQN_PPO="DQN",  # , "DDQN", "PPO"
        lags=14,
        gamma=0.5,
        memory_size=2000,
        batch_size=64,
        n_step=128,
        mini_batch_size=32,
        gae_lambda=0.98,
        gradient_clip_norm=10.,
        epochs=5,
        actor_lr=1e-4,
        critic_lr=5e-04,

):
    """
    定义超参,从
    1)数据生成;
    2)数据标准化;
    3)训练数据生成;验证数据生成;
    4)模型建模: DQN/DDQN/PPO;
    5)FinRobot_Agent建模
    以上诸个环节定义超参,再:
    6)训练
    7)验证数据生成的验证env,接受训练模型生成total reward;
    :args:
        horizon: int
        lookback: int
        normalize_columns: list
        DQN_DDQN_PPO: str "DQN", "DDQN", "PPO"
        lags: int
        gamma: float
        memory_size: int
        batch_size: int
        n_step: int
        mini_batch_size:
        gae_lambda = 0.98
        gradient_clip_norm = 10.
        epochs = 5

    :return: 验证数据,全部经模型策略后的total reward
    """
    # 调用BTC爬取部分
    sys.path.append("l:/Python_WorkSpace/量化交易/")  # 增加指定的绝对路径,进入系统路径,从而便于该目录下的库调用
    Folder_base = "l:/Python_WorkSpace/量化交易/data/"
    config_file_path = "l:/Python_WorkSpace/量化交易/BTCCrawl_To_DataFrame_Class_config.ini"
    URL = 'https://data.binance.com'
    StartDate = "2023-1-20"
    EndDate = "2023-09-10"
    BTC_json = "BTC_h12.json"
    BinanceBTC_json = "BinanceBTC_h12.json"
    api_key, api_secret = get_api_key(config_file_path)
    BTC_data = BTC_DataAcquire(URL, StartDate, EndDate, Folder_base, BTC_json,
                               binance_api_key=api_key, binance_api_secret=api_secret)
    data = BTC_data.MarketFactor_ClosePriceFeatures(by_BinanceAPI=True,
                                                    FromWeb=False, close_colName='close', lags=0, window=20,
                                                    horizon=horizon,
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
    split = np.argwhere(data_normalized.index == pd.Timestamp('2023-01-01', tz='UTC'))[0, 0]

    #########Arguments Optimization#############
    Pretrained_model = False

    lags = lags
    action_n = 3
    gamma = gamma
    memory_size = memory_size
    replay_batch_size = int(memory_size / 2)
    batch_size = batch_size
    DQN_episode = 80
    DDQN_episode = 80

    DQN_saved_model_filename = "230610-51"
    DDQN_saved_model_filename = "230630-32"
    # PPO部分
    n_worker = 8
    n_step = n_step
    mini_batch_size = int(n_worker * n_step / 4)  # int(n_worker * n_step / 4)
    gae_lambda = gae_lambda
    gradient_clip_norm = gradient_clip_norm
    epochs = epochs
    updates = 1200
    today_date = pd.Timestamp.today().strftime('%y%m%d')
    PPO_saved_model_filename = '230703-10'

    ####################

    if DQN_DDQN_PPO == "DQN" or DQN_DDQN_PPO == "DDQN":
        # 生成dataset, env, 建模
        dataset_train = Dataset_Generator(data_normalized[:split], lags=lags,
                                          data_columns_state=normalize_columns)
        dataset_test = Dataset_Generator(data_normalized[split:], lags=lags,
                                         data_columns_state=normalize_columns)
        # 训练environment, 测试environment:
        env = Finance_Environment_V2(dataset_train, dataset_type='tensorflow_Dataset', action_n=action_n,
                                     min_performance=0.,
                                     min_accuracy=0.0)  # 允许做空,允许大亏,使得更多的训练数据出现
        env_test = Finance_Environment_V2(dataset_test, dataset_type='tensorflow_Dataset', action_n=action_n,
                                          min_performance=0.,
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
            # saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)

            if Pretrained_model:  # 调出之前训练的模型,接续训练:
                ckpt = tf.train.Checkpoint(model=Q)
                saved_path = saved_path_prefix + DDQN_saved_model_filename
                ckpt.restore(
                    saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,

            # DDQN 训练过程:
            FinR_Agent_DDQN.learn(episodes=DDQN_episode)
            print(f"{'-' * 40}finished{'-' * 40}")
            model = Q
            action_strategy_mode = 'argmax'

        elif DQN_DDQN_PPO == 'DQN':  # DQN
            saved_path_prefix = 'saved_model/BTC_DQN_'
            # saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
            if Pretrained_model:  # 调出之前训练的模型,接续训练:
                ckpt = tf.train.Checkpoint(model=Q)
                saved_path = saved_path_prefix + DQN_saved_model_filename
                ckpt.restore(
                    saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,

            # DQN 训练过程:
            FinR_Agent_DQN.learn(episodes=DQN_episode)
            print(f"{'-' * 40}finished{'-' * 40}")
            model = Q
            action_strategy_mode = 'argmax'

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
                                     min_accuracy=0.0)  # 允许做空,允许大亏,使得更多的训练数据出现
        env_test = Finance_Environment_V2(iter_dataset_test, dataset_type='ndarray_iterator', action_n=action_n,
                                          min_performance=0.,
                                          min_accuracy=0.0)  # 允许做空,允许大亏,使得更多的训练数据出现
        _, obs_n = env.observation_space.shape

        # PPO 建模:
        saved_path_prefix = 'saved_model/BTC_PPO_'

        Actor = ActorModel(seq_len=lags, in_features=obs_n, out_features=action_n)
        Critic = CriticModel(seq_len=lags, in_features=obs_n, out_features=action_n)

        # 训练用 多进程 建模
        workers = []
        for i in range(n_worker):
            worker = Worker(dataset=iter_dataset_train, dataset_type='ndarray_iterator', action_dim=action_n)
            workers.append(worker)
        FinR_Agent_PPO = PPO2(workers, Actor, Critic, action_n, lags, obs_n, actor_lr=actor_lr, critic_lr=critic_lr,
                              gae_lambda=gae_lambda, gamma=gamma,
                              c1=1., gradient_clip_norm=gradient_clip_norm, n_worker=n_worker, n_step=n_step,
                              epochs=epochs, mini_batch_size=mini_batch_size)
        # saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)

        if Pretrained_model:  # 调出之前训练的模型,接续训练:
            ckpt = tf.train.Checkpoint(actormodel=Actor, criticmodel=Critic)
            saved_path = saved_path_prefix + PPO_saved_model_filename
            ckpt.restore(
                saved_path)  # 奇葩(搞笑)的是,这里的saved_path不能带.index的文件类型后缀,必须是完整的文件名不带文件类型后缀,

        # PPO 训练过程:
        FinR_Agent_PPO.nworker_nstep_training_loop(updates=updates)
        FinR_Agent_PPO.close_process()
        print(f"{'-' * 40}finished{'-' * 40}")
        model = Actor
        action_strategy_mode = 'tfp.distribution'

    BacktestEvent = BacktestingEventV2(env_test, model, initial_amount=1000, percent_commission=0.001,
                                       fixed_commission=0., verbose=False, MinUnit_1Position=-8, )
    BacktestEvent.backtest_strategy_WO_RM(action_strategy_mode=action_strategy_mode)

    return BacktestEvent.net_wealths


class FinRobotTuner(keras_tuner.RandomSearch):
    # class FinRobotTuner(keras_tuner.Hyperband):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        Backtest_wealth = FinRobotSearchSpace(
            horizon=hp.Int('horizon', min_value=1, max_value=10, step=1),
            lookback=hp.Choice('lookback', [225, 365, 730]),
            MarketFactor=hp.Boolean('MarketFactor'),
            DQN_DDQN_PPO="PPO",  # , "DDQN", "PPO"
            lags=hp.Choice('lags', [3, 5, 7, 14, 20]),
            gamma=hp.Choice('gamma', [.45,0.5, 0.6, 0.7, 0.8,.85, 0.9, 0.92, 0.95, 0.98]),
            memory_size=hp.Choice("memory_size", [32, 64, 256, 512, 1024, 2000]),  # PPO时,不需要
            batch_size=hp.Choice("batch_size", [16, 32]),
            n_step=hp.Choice("n_step", [1, 3, 5, 10, 20, 32, 64, 128]),
            gae_lambda=hp.Choice("gae_lambda", [0.8,0.85,0.9,.92, 0.94, 0.96, 0.98]),
            gradient_clip_norm=hp.Choice("gradient_clip_norm", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
            epochs=hp.Choice("epochs", [3, 5]),
            actor_lr=hp.Choice("actor_lr", [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]),
            critic_lr=hp.Choice("critic_lr", [5e-3, 1e-4, 5e-04, 1e-5, 5e-5]),
        )
        # Return a dictionary of metrics for KerasTuner to track.
        metrics_dict = {"net_wealth": Backtest_wealth["net_wealth"][-1]}  # 取最终的net_wealth
        return metrics_dict


def result_summary_DataFrame(path, best_num, save_path):
    """
    以pandas DataFrame形式列出超参按照score从大到小排列,打印,并以json文件格式存盘
    :param
        path: keras_tuner的project_name;
        best_num: 提取最好的超参的前几个;
        save_path: 转换成json文件后,存盘路径;(包含文件名.json)

    :return: 超参DataFrame,最后一列为score;
    """

    # 获取所有的trial.json文件
    trial_files = glob.glob('{}/trial_*/*.json'.format(path))
    # 解析trial.json文件并获取每个试验的超参数配置和评估结果
    trials = []
    for trial_file in trial_files:
        with open(trial_file, 'r') as f:
            trial = json.load(f)
            trials.append(trial)

    # 根据评估结果对试验进行排序
    sorted_trials = sorted(trials, key=lambda x: x['score'], reverse=True)

    # 将hyperparameters.values,首先获得字典,再转换成DataFrame:
    search_results_dict = [sorted_trials[i]['hyperparameters']['values'] for i in
                           range(min(best_num, len(sorted_trials)))]
    search_results_DF = pd.DataFrame(search_results_dict)
    # score再单独列出,放在最后DataFrame最后一列
    search_results_score = [sorted_trials[i]['score'] for i in range(min(best_num, len(sorted_trials)))]
    search_results_DF['score'] = search_results_score
    print(search_results_DF)
    if save_path is not None:
        search_results_DF.to_json(save_path)

    return search_results_DF


if __name__ == '__main__':
    # Random Search:
    # tuner = FinRobotTuner(
    #     # Objective is one of the keys.
    #     objective=keras_tuner.Objective("net_wealth", "max"),
    #     max_trials=50, overwrite=True, directory="saved_model", project_name="keras_tuner",
    # )
    # Hyperband Search: # 不知道为什么,hyperband 算法,会在执行到tuner.search()时,直接显示result summary,然后退出;
    # tuner = FinRobotTuner(
    #     # Objective is one of the keys.
    #     objective=keras_tuner.Objective("net_wealth", "max"),
    #     max_epochs=6, overwrite=True, directory="saved_model", project_name="keras_tuner",
    # )
    # tuner.search()
    # Retraining the model
    # search_result = tuner.results_summary()
    # print(search_result)

    save_path = r'l:/Python_WorkSpace/量化交易/FinanceRobot/saved_model/'
    read_path = f'{save_path}keras_tuner'
    best_num = 10
    save_path = '{}RandomSearch{}_PPO.json'.format(save_path, best_num, )
    result_summary = result_summary_DataFrame(read_path, best_num=best_num, save_path=save_path)

    # result_summary.to_csv('{}RandomSearch{}.csv'.format(path,best_num))
    # read_path = '{}RandomSearch{}.json'.format(path,best_num)
    # result_summary = pd.read_json(read_path,)

    # best_hp = tuner.get_best_hyperparameters()[0]
    best_hp = result_summary.iloc[0]  # 获得最佳的第一行;
    best_hp = best_hp.drop('score')  # 去除'score'列,因为score不属于FinRobot模型的参数;
    best_hp = best_hp.to_dict()  # 转换成字典;
    for key, value in best_hp.items():
        print(f"{key}={value}")
    # 使用best_hp来训练,训练次数为FinRobotSearchSpace中定义的DQN_episode,DDQN_episode,updates
    # FinRobotSearchSpace(**best_hp, DQN_DDQN_PPO='PPO',)
