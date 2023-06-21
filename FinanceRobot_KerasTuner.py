# 使用Keras.Tuner;构建搜索空间;参数优化;
import warnings

warnings.filterwarnings('ignore')

import sys
import tensorflow as tf
import keras_tuner

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
from FinanceRobot_DDQNPPOModel_lib import Decompose_FF_Linear, FinRobotAgentDQN, FinRobotAgentDDQN
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
    sys.path.append("e:/Python_WorkSpace/量化交易/")  # 增加指定的绝对路径,进入系统路径,从而便于该目录下的库调用
    Folder_base = "e:/Python_WorkSpace/量化交易/data/"
    config_file_path = "e:/Python_WorkSpace/量化交易/BTCCrawl_To_DataFrame_Class_config.ini"
    URL = 'https://data.binance.com'
    StartDate = "2023-1-20"
    EndDate = "2023-06-10"
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
    # Test_flag = False
    # train_test_text_add = 'test' if Test_flag else 'train'

    lags = lags
    action_n = 3
    gamma = gamma
    memory_size = memory_size
    replay_batch_size = int(memory_size / 2)
    batch_size = batch_size
    DQN_episode = 20
    DDQN_episode = 20
    # PPO部分
    n_worker = 8
    n_step = n_step
    mini_batch_size = int(n_worker * n_step / 4)  # int(n_worker * n_step / 4)
    gae_lambda = gae_lambda
    gradient_clip_norm = gradient_clip_norm
    epochs = epochs
    updates = 50
    today_date = pd.Timestamp.today().strftime('%y%m%d')

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
            # saved_path_prefix = 'saved_model/BTC_DDQN_Tunner'
            # saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
            # DDQN 训练过程:
            FinR_Agent_DDQN.learn(episodes=DDQN_episode)
            print(f"{'-' * 40}finished{'-' * 40}")
            model = Q
            action_strategy_mode = 'argmax'

        elif DQN_DDQN_PPO == 'DQN':  # DQN
            # saved_path_prefix = 'saved_model/BTC_DQN_'
            # saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
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
        FinR_Agent_PPO = PPO2(workers, Actor, Critic, action_n, lags, obs_n, actor_lr=1e-4, critic_lr=5e-04,
                              gae_lambda=gae_lambda,
                              gamma=gamma,
                              c1=1., gradient_clip_norm=gradient_clip_norm, n_worker=n_worker, n_step=n_step,
                              epochs=epochs,
                              mini_batch_size=mini_batch_size)
        saved_path = '{}gamma0{}_lag{}_{}.h5'.format(saved_path_prefix, str(int(gamma * 100)), lags, today_date)
        # PPO 训练过程:
        FinR_Agent_PPO.nworker_nstep_training_loop(updates=updates)
        FinR_Agent_PPO.close_process()
        print(f"{'-' * 40}finished{'-' * 40}")
        model = Actor
        action_strategy_mode = 'tfp.distribution'

    BacktestEvent = BacktestingEventV2(env_test, model, initial_amount=1000, percent_commission=0.001,
                                       fixed_commission=0., verbose=True, MinUnit_1Position=-8, )
    BacktestEvent.backtest_strategy_WO_RM(action_strategy_mode=action_strategy_mode)

    return BacktestEvent.net_wealths


class FinRobotTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        Backtest_wealth = FinRobotSearchSpace(
            horizon=hp.Int('horizon', min_value=2, max_value=15, step=2),
            lookback=hp.Choice('lookback', [225, 365, 730]),
            MarketFactor=hp.Boolean('MarketFactor'),
            DQN_DDQN_PPO="DDQN",  # , "DDQN", "PPO"
            lags=hp.Choice('lags', [5, 7, 14, 20]),
            gamma=hp.Choice('gamma', [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.98]),
            memory_size=hp.Choice("memory_size", [32, 64, 256, 512, 1024, 2000]),
            batch_size=hp.Choice("batch_size", [16, 32]),
            n_step=hp.Choice("n_steps", [5, 10, 20, 32, 64, 128, 256]),
            gae_lambda=hp.Choice("gae_lambda", [0.96, 0.97, 0.98]),
            gradient_clip_norm=hp.Choice("gradient_clip_", [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
            epochs=hp.Choice("epochs", [3, 5])
        )
        return Backtest_wealth


if __name__ == '__main__':
    tuner = FinRobotTuner(
        max_trials=3, overwrite=True, directory="saved_model", project_name="keras_tunner",
    )
    tuner.search()
    # Retraining the model
    search_result = tuner.search_space_summary()
    print(search_result)
    # best_hp = tuner.get_best_hyperparameters()[0]
    # keras_code(**best_hp.values, saving_path="/tmp/best_model")
