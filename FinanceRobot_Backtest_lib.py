#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
# from collections import deque, namedtuple
# from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras.layers import Dense, Dropout, AveragePooling1D
import random
# import time

# import math
from tqdm import tqdm

import numpy as np
import pandas as pd


# Dataset生成函数:
def gen(data):
    """
    交易数据生成器;
    args:
        data: 为ndarray,非DataFrame;
    """
    for i in range(data.shape[0]):
        yield data[i, :]


def gen_date(date_list):
    """
    args:
        date_list: 数据的日期列表;如果是DataFrame.index,其为datetime64类型,需要转换成string类型输入,即date_list事先需要执行data_list.astype('string');
    out:
        交易日期的生成器
    """
    date_list = np.array(date_list)
    length = date_list.shape[0]
    for i in range(length):
        yield date_list[i]


def Dataset_Generator(data, data_columns_state=None, data_columns_non_state=None, lags=20, shuffle=False,
                      buffer_size=10000):
    """
    从交易数据Dataframe,生成dataset;
    args:
        data: Dataframe格式的数据(N,features);
        data_columns:Data的列的数量(包括features,以及不转入state_space的列,如horizon_log_return,close);
        lags: 样本延时的数量,即dataset window的大小; lags>=1;
        batch_size: 必须为1 ; 因为交易数据送入Finance_Environment后,需要每步输入一个动作,每step输出state,reward,done,info,故而batch_size,这里只能设为1
        shuffle: bool ; 是否shuffle;保持原交易数据的顺序,所以,不能shuffle;缺省为False
    out:
        xs: xs.shape:(date,(N,lags,state_features),(N,lags,non_state_features));元组,包含date(字符串),以及data_state,data_non_state;
            date字符串记录交易日的时间戳(例如: "2022-11-19 12:00:00")
    """
    # 设置data_columns_state, data_columns_non_state缺省值:
    if data_columns_state is None:
        data_columns_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if data_columns_non_state is None:
        data_columns_non_state = [-2, -1]
    data_state = data[data.columns[data_columns_state]]
    data_non_state = data[data.columns[data_columns_non_state]]

    output_signature_state = tf.TensorSpec((len(data_columns_state),), tf.float32)
    output_signature_non_state = tf.TensorSpec((len(data_columns_non_state),), tf.float32)

    xs_state = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature_state,
        args=(data_state,),
    )  # args用于给gen传递参数,必须是元组的形式传入参数;

    xs_non_state = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature_non_state,
        args=(data_non_state,),
    )  # args用于给gen传递参数,必须是元组的形式传入参数;

    date_list = data.index.astype('string')
    date = tf.data.Dataset.from_generator(
        gen_date,
        output_signature=tf.TensorSpec((), tf.string),
        # date_list如果是datetime64类型,需要提前转化string: date_list.astrype('string')
        args=(date_list,),
    )  # args用于给gen传递参数,必须是元组的形式传入参数;

    xs_state = xs_state.window(lags, shift=1, drop_remainder=True)
    xs_state = xs_state.flat_map(lambda w: w.batch(lags, drop_remainder=True))

    xs_non_state = xs_non_state.window(lags, shift=1, drop_remainder=True)
    xs_non_state = xs_non_state.flat_map(lambda w: w.batch(lags, drop_remainder=True))

    date = date.window(lags, shift=1, drop_remainder=True)
    date = date.flat_map(lambda w: w.batch(lags, drop_remainder=True))

    dataset = tf.data.Dataset.zip((date, xs_state, xs_non_state))
    if shuffle:
        dataset.shuffle(buffer_size)

    return dataset.batch(1, drop_remainder=True).prefetch(1)  # batch_size = 1


def data_normalization(data, lookback=252,
                       normalize_columns=None):
    """
    将data (N,features),进行函数变换,实现每个mean=0,std=1的标准化;
    利用tf.keras.layers.Normalization().adapt()方法,获取每个ds元素的mean与std,
    然后将,ds每个元素的最后一个数值,标准化成新的数值;
    :param:
        data: pandas, (N,features);
        lookback: int, 往回lookback的t时刻;在lookback的时间段内,采样mean,std,以将lookback的最后时刻数据标准化;
        normalize_columns: data的特征列中,需要normalization的列的list;缺省=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
        最后两列,'horizon_price','close'不参与标准化
        Binance采集的Data列可能为:
        ['log_return', 'Roll_price_sma', 'Roll_price_min', 'Roll_price_max',
       'Roll_return_mom', 'Roll_return_std', 'volume', 'RSI_7',
       'Log_close_weeklag', 'Log_high_low', 'Log_open_weeklag',
       'open_pre_close', 'high_pre_close', 'low_pre_close', 'num_trades','bid_volume',
       "log_return_unnormalized",'horizon_price', 'close']
    :return:
        data: pandas, (N-lookback,features);
    """
    # 定义normalize_columns 缺省值:
    if normalize_columns is None:
        normalize_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    data_tobenormalized = data[data.columns[normalize_columns]]
    # axis=0, across the row; window为time_offset时,需要时固定的时间长度,1Y是不确定的长度
    rolling_mean = data_tobenormalized.rolling(window=lookback, min_periods=lookback, axis=0).mean()
    rolling_std = data_tobenormalized.rolling(window=lookback, min_periods=lookback, axis=0).std()
    rolling_mean.dropna(inplace=True)  # 去除N/A,即,去除window长度不到lookback的数据行;
    rolling_std.dropna(inplace=True)  # 去除N/A,即,去除window长度不到lookback的数据行;
    data_tobenormalized = data_tobenormalized.loc[rolling_mean.index]
    data_tobenormalized = (data_tobenormalized - rolling_mean) / (rolling_std + 1e-8)

    data = data.loc[data_tobenormalized.index]
    data[data.columns[normalize_columns]] = data_tobenormalized

    return data


# 金融沙箱- 模仿OpenAI GYM环境,创建Fiance类, 即,交易市场环境

class observation_space:
    """
    args:
        lags: 样本(交易时刻)延时的个数,表示几个延时交易时刻为一组;
        n_features: 每个交易时刻样本数据的特征数;注意,仅仅送入训练的state_feature,不包含non_state_feature
    """

    def __init__(self, lags, n_features):
        self.shape = (lags, n_features)


class action_space:
    """
    args:
        n: action的维度,每个观察状态(时刻)可以采取的各类action的种类
    """

    def __init__(self, n):
        self.n = n

    def sample(self):
        """
        随机输出action的值,从0到n-1
        """
        return random.randint(0, self.n - 1)  # random.randint()是两端都包含的随机值


class Finance_Environment:
    """
    类OpenAI Gym的environment,实现交易state,action,reward,next_state;
    action_space: 2; buy or sell;
    state_space: (lags,features);
    reward function: log_return > 0 , reward = 1 + abs(log_return * leverage);
                     log_return < 0, reward = 0 - abs(log_return * leverage);

    不允许做空;
    """

    def __init__(
            self,
            dataset,
            leverage=1,
            min_performance=0.85,
            min_accuracy=0.5,
    ):
        """
        args:
            dataset: Finance Environment的交易数据,为tf.data.Dataset类型,shape:((N,date),(N,lags,features))
            (定义每个lags的最后一个时序,表示当前状态所对应的时序.这是为了能够取得最后时序的模型预测值.)
        """

        self.dataset = dataset
        # 定义和初始化迭代器,内装Dataset,指针从第一个数据开始,为不影响指针,该变量仅在next()时使用.
        self.iter_dataset = iter(self.dataset)
        self.batch_size, self.lags, self.features = iter(dataset).element_spec[1].shape
        self.dataset_len = len(list(iter(dataset)))  # dataset的样本总条数(batch之后);

        self.leverage = leverage  # 杠杆
        self.min_performance = min_performance
        self.min_accuracy = min_accuracy
        self.observation_space = observation_space(self.lags, self.features)
        self.action_space = action_space(2)  # 假定涨跌两个动作;

    def dataset2data(
            self,
            price_Scaler=None,
            log_return_Scaler=None,
            price_column=-1,
            log_return_column=0,
            Mark_return_column=6,
    ):
        """
        将environment的dataset,还原出numpy类型的data:包括列:date,log_return,Mark_return,close,
        dataset数据如果曾经归一化,则需要输入归一化的Scaler(sklearn归一化模型类),来还原数据原值;缺省值None,则不需要归一化还原.
        args:
            price_Sclaer: 收盘价close归一化的Scaler;
            log_return_Scaler: log_return归一化的Scaler;
            price_column: price列在dataset中的列序列号,缺省为-1,即最后1列;
            log_return_column: log_return列在dataset中的列序列号,缺省为0,即第0列;
            Mark_return_column: Mar_return列在dataset中的列序列号,缺省为6,即第6列;
        out:
            env_data: 包含0)date(datetime64类型),1)log_return,2)Mark_return,3)close收盘价,numpy,shape(N,4)
        """
        date = np.array(
            [], dtype=object
        )  # object类型,才可以与数值合并到一个array里面;(datetime64类型不行)
        log_return = []
        Mark_return = []
        price = []

        for day, Data in tqdm(
                self.dataset,
                total=self.dataset_len,
        ):
            # 1.date:
            day = day[0, -1].numpy()  # 读取date str; lags=-1最后一个时序
            day = np.datetime64(day)  # 字符串转化成datetime64类型;
            date = np.append(date, day)
            # 2. log_return
            logReturn = Data[0, -1, log_return_column]  # lags=-1,log_return第0列
            if log_return_Scaler is not None:
                logReturn = log_return_Scaler.inverse_transform(
                    np.array(logReturn).reshape(1, -1)
                )  # 归一化之后的还原;
            log_return = np.append(log_return, logReturn)

            # 3. Mark_return:
            # lags=-1,Mark_return倒数第2列,顺数第6列; 该列归一化前后数值(0或1)无变化
            Mark = Data[0, -1, Mark_return_column]
            Mark_return = np.append(Mark_return, Mark)
            # 4.price:
            price_step = Data[0, -1, price_column]  # lags=-1最后一个时序
            if price_Scaler is not None:
                price_step = price_Scaler.inverse_transform(
                    np.array(price_step).reshape(1, -1)
                )  # 归一化之后的还原;
            price = np.append(price, price_step)

            # 合并 date,log_return,Mark_return,price
        env_data = np.c_[
            date.reshape(-1, 1),
            log_return.reshape(-1, 1),
            Mark_return.reshape(-1, 1),
            price.reshape(-1, 1),
        ]
        return env_data  # (N,4)
        # return date,log_return,Mark_return,price

    def get_state(self, bar):
        # Dataset类型,获得Dataset中,指定序列号的的element, ((N,date),(N,lags,features))
        element = [d for i, d in enumerate(iter(self.dataset)) if i == bar][0]
        # 此处待查,疑问有2: a)列表表达式内元素,是否应该就是((N,date),(N,lags,features)),那么[][0]代表什么呢? 哦, 找到的第0个值,待确认;
        # 疑问2: b)if i == bar 寻找到的仅仅是,self.dataset内某个batch的序列号,是否确认是某一条交易数据呢?
        state = element[1]  # (N,lags,features)
        return state

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def reset(self):
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.iter_dataset = iter(self.dataset)  # 复位dataset迭代器,使其从第一个数据开始.
        self.bar = 0
        self.state = self.iter_dataset.next()[1]  # (1,lags,features) 第一个数据
        return self.state

    def step(self, action):
        """
        state: 观察的状态,或者从reset而来,或者从上一个step而来; (N,lags,features);
        action: 观察的状态的策略action;
        """
        # correct = action == self.data['d'].iloc[self.bar]
        # ((N,date),(N,lags,features))->(features,)

        # 首列为log_return;取lags的最后一个时序表示当前状态;这里的log_return经过归一化之后数值范围为[0,1]
        log_return = self.state[0, -1, 0]
        # Mark_return列在倒数第二列,顺数第6列; lags取第0个, MinmaxScale之后,变成float32了,需要转回成int
        Mark_return = int(self.state[0, -1, 6])

        correct = action == Mark_return
        # ret = self.data['r'].iloc[self.bar] * \
        #     self.leverage  # r:bar时刻与上一个时刻的比值的对数再乘以杠杆率,反应的是收益值;
        # 相对于上一个交易日,reward为1,表明是对上一个action的reward,因此,最佳策略的action,也就是下一个reward近期收益+后续所有时刻的远期收益之和;
        return_step = log_return * self.leverage
        reward_1 = 1 if correct else 0
        # 相对于前日盈利,则奖励按杠杆率加权;亏损,则惩罚也同比按杠杆率加权
        reward_2 = abs(return_step) if correct else -abs(return_step)
        self.treward += reward_1  # 表示的是,当action=1,即策略认为产生正收益action的执行次数;
        self.bar += 1
        self.accuracy = self.treward / self.bar  # accurancy: 正确决策action的比例;
        # performance: 以 1*exp(reward_2),反应的是总收益率
        self.performance *= np.exp(reward_2)
        # if self.bar >= self.dataset_len-1:  # bar从0->dataset_len,不包括dataset_len
        if self.bar >= self.dataset_len:  # bar=dataset_len,实际是最后的一条数据的下一条;
            done = True
            # 当走到最后一条样本时,dataset迭代器已经不能够再next(),否则会出错,这里直接结束;输出的state为原state
            info = {}
            return self.state, reward_1 + reward_2 * 5, done, info
        elif reward_1 == 1:
            done = False
        elif (
                self.performance < self.min_performance and self.bar > 15
        ):  # 当最佳策略reward=0,总收益率低于最小值,步数(时刻)大于(15)次时,结束;即,执行策略action(15)次以后,收益率仍小于最小收益率,action策略仍然是显亏损,就中止结束
            done = True
        elif (
                self.accuracy < self.min_accuracy and self.bar > 15
        ):  # 当执行(15)步以后,产生收益的action的次数,仍低于最小次数时,结束中止,停止训练;
            done = True
        else:
            done = False

        self.state = next(self.iter_dataset)[1]  # 下一个state

        info = {}
        return self.state, reward_1 + reward_2 * 5, done, info  # reward_2*5 不知道为什么放大5倍


# 重写基于Finance类Env的Finance Agent,实现改进了的Agent在模拟的交易环境中,使用各强化学习算法,逐次从历史交易数据中学习正确的策略动作;
class Finance_Environment_V2:
    """
    类OpenAI Gym的environment,实现交易state,action,reward,next_state;
    action_space: 3; sell=0 , hold=1, buy=2 ;
    state_space: (lags,features);
    reward function: log_return > 0 , reward = 1 + (log_return * leverage) * (action -1 ) - trading_cost * abs(action - 1);
                     log_return < 0, reward = 0 + (log_return * leverage) * (action -1 ) - trading_cost * abs(action - 1);

    训练时允许做空,回测时不允许做空;
    """

    def __init__(
            self,
            dataset,
            action_n,
            leverage=1,
            trading_commission=0.002,
            min_performance=0.3,  # 允许做空,
            min_accuracy=0.3,
    ):
        """
        args:
            dataset: Finance Environment的交易数据,为tf.data.Dataset类型,shape:((1,date),(1,lags,state_features),(1,lags,non_state_features))
            (定义每个lags的最后一个时序,表示当前状态所对应的时序.这是为了能够取得最后时序的模型预测值.)
        """

        self.dataset = dataset
        # 定义和初始化迭代器,内装Dataset,指针从第一个数据开始,为不影响指针,该变量仅在next()时使用.
        self.iter_dataset = iter(self.dataset)
        self.batch_size, self.lags, self.features = iter(dataset).element_spec[1].shape
        self.dataset_len = len(list(iter(dataset)))  # dataset的样本总条数(batch之后) batch_size =1 ;

        self.leverage = leverage  # 杠杆
        self.trading_commission = trading_commission
        self.min_performance = min_performance
        self.min_accuracy = min_accuracy
        self.observation_space = observation_space(self.lags, self.features)
        self.action_space = action_space(action_n)  # buy=2,hold=1,sell=0;

        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.bar = 0

    def dataset2data(
            self,
            price_column=-1,
            horizon_price_column=-2,
    ):
        """
        将environment的dataset,还原出numpy类型的data:包括列:date,log_return,Mark_return,close,
        dataset数据如果曾经归一化,则需要输入归一化的Scaler(sklearn归一化模型类),来还原数据原值;缺省值None,则不需要归一化还原.
        args:
            price_column: price列在dataset中的列序列号,缺省为-1,即最后1列;
            log_return_column: log_return列在dataset中的列序列号,缺省为-2,即第-2列;
        out:
            env_data: 包含0)date(datetime64类型),1)horizon_price,2)close收盘价,numpy,shape(N,3)
        """
        pass  # 数据标准化后,需要重写

    def _get_state(self, bar):
        # Dataset类型,获得Dataset中,指定序列号的的element, ((N,date),(N,lags,state_features),(N,lags,non_state_features))
        element = [d for i, d in enumerate(iter(self.dataset)) if i == bar][0]
        # 此处待查,疑问有2: a)列表表达式内元素,是否应该就是((N,date),(N,lags,features)),那么[][0]代表什么呢? 答案: 找到的第0个值,待确认;
        # 疑问2: b)if i == bar 寻找到的仅仅是,self.dataset内某个batch的序列号,是否确认是某一条交易数据呢?
        # 答案: iter(self.dataset),即是迭代出包含的每一个元素,无论batch_size;
        date, state, non_state = element  # (1,date),(1,lags,state_features),(1,lags,non_state_features)
        return date, state, non_state

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def reset(self):
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.iter_dataset = iter(self.dataset)  # 复位dataset迭代器,使其从第一个数据开始.
        self.bar = 0
        _, self.state, self.non_state = self.iter_dataset.next()  # (1,date),(1,lags,state_features),(1,lags,non_state_features)

        return self.state, self.non_state

    def step(self, action):
        """
        state: 观察的状态,或者从reset而来,或者从上一个step而来; (N,lags,state_features);
        action: 观察的状态的策略action;
        """

        # non_state: (N,lags,non_state_features), 包括未曾标准化/归一化的,'horizon_price','close';
        horizon_price, current_price = self.non_state[0, -1, :]
        # trading_cost = self.trading_commission * np.log(current_price)
        info = {'bar': self.bar,
                'price': current_price,
                'horizon_price': horizon_price
                }

        if self.bar < self.dataset_len and not tf.experimental.numpy.isnan(horizon_price):
            done = False
            reward_1 = int(horizon_price > current_price) * (action - 1)
            # reward_1 = 1 if horizon_log_return > 0 else -1
            # 相对于前日盈利,则奖励按杠杆率加权;亏损,则惩罚也同比按杠杆率加权
            # reward_2 = (horizon_log_return * self.leverage) * (action - 1)
            # 收益-佣金: [horizon_price - horizon_price * commission * (action - 1) ] / price
            reward_2 = np.log(horizon_price * (1 - self.trading_commission * abs(action - 1)) / current_price)
            # 这里exp(horizon_log_return)就是horizon时刻的收盘价,后面直接替代
            # trading_cost = trading_cost * abs(action - 1)
            # reward = reward_1 + reward_2 * 5 - trading_cost * 5
            reward = reward_1 + reward_2 * 5
            self.treward += reward_1  # 表示的是,当action=1,即策略认为产生正收益action的执行次数;
            self.accuracy = self.treward / (self.bar + 1)  # accuracy: 正确决策action的比例;
            # performance: 以 1*exp(reward_2),反应的是总收益率
            self.performance *= np.exp(reward_2)

            _, self.state, self.non_state = next(self.iter_dataset)  # 下一个state

            # 当最佳策略reward=0,总收益率低于最小值,步数(时刻)大于(15)次时,结束;即,执行策略action(15)次以后,收益率仍小于最小收益率,action策略仍然是显亏损,就中止结束
            if self.performance < self.min_performance and self.bar > 15:
                done = True
            # 当执行(15)步以后,产生收益的action的次数,仍低于最小次数时,结束中止,停止训练;
            if self.accuracy < self.min_accuracy and self.bar > 15:
                done = True
        else:
            done = True
            reward = 0

        self.bar += 1

        return self.state, reward, done, info

    # agent的向量化backtesting,注意的是:


# 其就训练后的模型,做出了每个样本的预测,根据预测后的action_Qvalue队,预测了策略action,根据action,增加了一列头寸


def Backtesting_vector(
        agent_model,
        env,
        price_Scaler=None,
        log_return_Scaler=None,
        price_column=-1,
        log_return_column=0,
        Mark_return_column=6,
):
    """
    arg:
        agent_model: agent中训练后的model,这样预测出的才有backtest的意义;
        env: 为类OpenAI的Finance environment; 包含有backtest的观察dataset;
        environment的dataset数据如果曾归一化,则需要输入归一化的Scaler(sklearn归一化模型类),来还原数据原值;缺省值None,则不需要归一化还原.
        price_Scaler: 收盘价归一化的Scaler;
        log_return_Scaler: log_return归一化的Scaler
        price_column: price列在dataset中的列序列号,缺省为-1,即最后1列;
        log_return_column: log_return列在dataset中的列序列号,缺省为0,即第0列;
        Mark_return_column: Mar_return列在dataset中的列序列号,缺省为6,即第6列;
    out:
        data: shape(N,6),包括列: 0)date,1)log_return,2)Mark_return,3)price,4)strategy_return,5)position
        ;以及累计求和的散件图
    """
    env.min_accuracy = 0.0
    env.min_performance = 0.0

    env_data = env.dataset2data(
        price_Scaler=price_Scaler,
        log_return_Scaler=log_return_Scaler,
        price_column=price_column,
        log_return_column=log_return_column,
        Mark_return_column=Mark_return_column,
    )

    # done = False
    state = env.reset()
    positions = np.zeros((env.dataset_len,), dtype=int)
    for _ in tqdm(range(env.dataset_len)):  # dataset中最后一个data,没有next step;
        # 1.positions
        action = np.argmax(agent_model.predict(state)[0, 0])
        position = 1 if action == 1 else -1
        positions[_] = position
        state, reward, done, info = env.step(action)

        if done:
            break

    # strategy_return = np.roll(positions, 1)*env.leverage *env_data[:, 1]  # data[:,1]:log_return
    # print('positions.shape:{};env_data[:,1].shape:{}'.format(positions.shape,env_data[:,1].shape))
    strategy_return = (
            positions * env.leverage * env_data[:, 1]
    )  # 没有乘以价格基数,日收益率如何反映真实的收益?因为时刻连续,时刻累计
    env_data = np.c_[env_data, strategy_return, positions]  # (N,6)
    return env_data


# 基于事件的回测 Event_based Backtest


class Backtesting_event:
    """
    Event based Backtesting
    """

    def __init__(
            self,
            env,
            model,
            amount,
            percent_commission,
            fixed_commission,
            verbose=False,
            price_Scaler=None,
            log_return_Scaler=None,
            MinUnit_1Position=0,
    ):
        """
        args:
            env: 类OpenAI GYM的Finance 类环境,由Fiance_environment类生成;给定state,以及action,能够生成next_state,reward;
            model: 训练后的DQN模型,能够针对state给出策略action;
            amount: 回测的初始金额;
            percent_commission: transaction时的与价格相比例的交易手续费;
            fixed_commission: transaction时,一次性,固定的交易手续费
            price_Scaler: 收盘价归一化的Scaler;
            log_return_Scaler: log_return归一化的Scaler
            MinUnit_1Position:头寸的最小计量单位;对于股票而言,最小的单位是整数,即为10的0次方,取值0;对于BTC而言,最小的单位是10的-8次方,即取-8;如果某只股票最小单位为10,则值为1;
        """
        self.env = env
        self.model = model
        self.initial_amount = amount
        self.current_balance = amount
        self.ptc = percent_commission  # 按比例形式的交易佣金;percent transaction commission
        self.ftc = fixed_commission  # 固定形式的交易佣金; fixed transaction commission
        self.verbose = verbose
        self.units = 0
        self.trades = 0
        self.net_performance = 0
        self.env_data = env.dataset2data(
            price_Scaler=price_Scaler, log_return_Scaler=log_return_Scaler
        )
        self.MinUnit_1Position = MinUnit_1Position

    def get_date_price(self, bar):
        """Returns date and price for a given bar."""
        date = self.env_data[bar, 0]
        price = self.env_data[bar, -1]
        return date, price

    def print_balance(self, bar):
        """Prints the current cash balance for a given bar."""
        date, price = self.get_date_price(bar)
        print(f"{date} | current balance = {self.current_balance:.2f}")

    def calculate_net_wealth(self, price):
        return self.current_balance + self.units * price

    def print_net_wealth(self, bar):
        """Prints the net wealth for a given bar
        (cash + position).
        """
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        # print(f'{date} | net wealth = {net_wealth:.2f}')
        print(
            "{} | balance:{:.2f}+units:{}*price:{:.4f}=net_wealth:{:.4f}".format(
                date, self.current_balance, self.units, price, net_wealth
            )
        )

    def set_prices(self, price):
        """Sets prices for tracking of performance.
        To test for e.g. trailing stop loss hit.
        """
        self.entry_price = price
        self.min_price = price
        self.max_price = price

    def place_buy_order(self, bar, amount=None, units=None, gprice=None):
        """Places a buy order for a given bar and for
            a given amount or number of units. 当units<0,表示买空;
        args:
            gprice: 买单的指定价格,guarantee价格;
        """
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            MinUnit_1Position = 10 ** self.MinUnit_1Position
            units = (
                    int(amount / price / MinUnit_1Position) * MinUnit_1Position
            )  # 获得低于指定位数小数的值;
            # print('units({})=int(amount({})/price({}))'.format(units, amount, price))
            # units = amount / price  # alternative handling
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc
        self.units += units
        self.trades += 1
        self.set_prices(price)
        if self.verbose:
            # print(f'{date} | buy {units} units for {price:.4f}')
            print(
                "{}'s price {:0.4f}, buy {} units.".format(
                    date,
                    price,
                    units,
                )
            )
            self.print_balance(bar)

    def place_sell_order(self, bar, amount=None, units=None, gprice=None):
        """Places a sell order for a given bar and for
            a given amount or number of units.
        args:
            gprice: 卖单的指定价格,guarantee价格;
        """
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            MinUnit_1Position = 10 ** self.MinUnit_1Position
            units = (
                    int(amount / price / MinUnit_1Position) * MinUnit_1Position
            )  # 获得低于指定位数小数的值;
            # units = amount / price  # altermative handling
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        self.trades += 1
        self.set_prices(price)
        if self.verbose:
            print("{}'s price {:0.4f}, sell {} units.".format(date, price, units))
            self.print_balance(bar)

    def close_out(self, bar):
        """Closes out any open position at a given bar."""
        date, price = self.get_date_price(bar)
        print(50 * "=")
        print(f"{date} | *** CLOSING OUT ***")
        if self.units < 0:
            self.place_buy_order(bar, units=-self.units)
        else:
            self.place_sell_order(bar, units=self.units)
        if not self.verbose:
            print(f"{date} | current balance = {self.current_balance:.2f}")
        self.net_performance = (self.current_balance / self.initial_amount - 1) * 100
        print(f"{date} | net performance [%] = {self.net_performance:.4f}")
        print(f"{date} | number of trades [#] = {self.trades}")
        print(50 * "=")

    def backtest_strategy_WO_RM(self):
        """
        Event-based backtesting of the trading bot's performance.
        利用BacktesingBase类中定义的交易的方法,买,卖等;实现观察数据的回测过程.
        没有risk Management;(没有止损,跟踪止损,止盈,风控措施)
        """
        self.units = 0
        self.position = 0  # 用于存放头寸的状态;
        self.trades = 0
        self.current_balance = self.initial_amount
        self.net_wealths = list()

        state = self.env.reset()
        action = np.argmax(self.model.predict(state)[0, 0])
        state, reward, done, info = self.env.step(action)  # 因为bar从1开始,state从0跳到1
        # bar不是从0开始,而是从1开始,是因为,初始第一条是第0时刻的当日收盘价,没有上个收盘价,无法下单;下单应从第1个时刻开始;
        for bar in range(1, self.env_data.shape[0]):
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * "=")
                print(f"{date} | *** START BACKTEST ***")
                self.print_balance(bar)
                print(50 * "=")
            action = np.argmax(self.model.predict(state)[0, 0])
            state, reward, done, info = self.env.step(action)
            position = 1 if action == 1 else -1
            if self.position in [0, -1] and position == 1:
                if self.verbose:
                    print(50 * "-")
                    print(f"{date} | *** GOING LONG ***")
                if (
                        self.position == -1
                ):  # 买单的价格是前一交易日的收盘价,所以bar-1,意味着买单的价格; #self.position=-1,表明手中有空头,先买空头,再买多头
                    self.place_buy_order(bar - 1, units=-self.units, gprice=None)
                self.place_buy_order(
                    bar - 1, amount=self.current_balance, gprice=None
                )  # 先买空头,再买多头;
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1:
                if self.verbose:
                    print(50 * "-")
                    print(f"{date} | *** GOING SHORT ***")
                if self.position == 1:
                    # 为什么是bar-1,前一日呢? 因为,place_sell_order方法是以指定样本的价格来买,该样本的价格是收盘价,所以是上一个样本的时间
                    self.place_sell_order(bar - 1, units=self.units, gprice=None)
                self.place_sell_order(
                    bar - 1, amount=self.current_balance, gprice=None
                )  # 手中已经卖出全部头寸,只有现金,再卖空(先借头寸,再卖出空头,卖出空头,头寸的余额为负值)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1
            # 以下输出资产/交易状态:

            self.net_wealths.append(
                (
                    date,
                    price,
                    self.units,
                    self.current_balance,
                    self.calculate_net_wealth(price),
                    self.position,
                    self.trades,
                )
            )
        self.net_wealths = pd.DataFrame(
            self.net_wealths,
            columns=[
                "date",
                "price",
                "units",
                "balance",
                "net_wealth",
                "position",
                "trades",
            ],
        )
        self.net_wealths.set_index("date", inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
        self.close_out(bar)

    def backtest_strategy_WH_RM(
            self,
            StopLoss=None,
            TrailStopLoss=None,
            TakeProfit=None,
            wait=5,
            guarantee=False,
    ):
        """Event-based backtesting of the trading bot's performance.
            Incl. stop loss, trailing stop loss and take profit.
            利用BacktesingBase类中定义的交易的方法,买,卖等;实现观察数据的回测过程.
            带有risk Management;包含:止损,跟踪止损,止盈等风控措施;
        args:
            StopLoss: stop loss,止损;
            TrailStopLoss: trailing stop loss,跟踪止损;
            TakeProfit: take profit,止盈;
            wait: 两次策略交易事件(风控事件,或者买卖交易)之间等待的条数(样本条数,或者是样本间隔交易间隔的数量)
            guarantee: bool值,表示是否以保证价格,或是市场当时的价格成交;
        """
        self.units = 0
        self.position = 0
        self.trades = 0
        self.sl = StopLoss
        self.tsl = TrailStopLoss
        self.tp = TakeProfit
        self.wait = 0
        self.current_balance = self.initial_amount
        self.net_wealths = list()

        state = self.env.reset()
        action = np.argmax(self.model.predict(state)[0, 0])
        state, reward, done, info = self.env.step(action)  # bar从1开始,跳到第1个样本;
        for bar in range(1, self.env_data.shape[0]):
            self.wait = max(0, self.wait - 1)
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * "=")
                print(f"{date} | *** START BACKTEST ***")
                self.print_balance(bar)
                print(50 * "=")

            # stop loss order
            if self.sl is not None and self.position != 0:  # 定义了止损,并且已有头寸,无论空,还是多
                # 根据最后一笔交易的进入价格(持有头寸的买卖价格),计算收益
                rc = (price - self.entry_price) / self.entry_price
                # 已有多头头寸,在此交易日(bar),该头寸的持有收益亏损超过设置的止损率.(1->2倍的ATR)
                if self.position == 1 and rc < -self.sl:
                    print(50 * "-")
                    if guarantee:
                        price = self.entry_price * (1 - self.sl)  # 成交价格设置成指定的止损价格;
                        print(f"*** STOP LOSS (LONG  | {-self.sl:.4f}) ***")
                    else:  # 否则,就是交易日的当时价格成交; 当时的价格可能时动态的,这里采用当日的收盘价成交;亦即,假设在当日收盘价之后,下一日到来之前交易;
                        print(f"*** STOP LOSS (LONG  | {rc:.4f}) ***")
                    self.place_sell_order(bar, units=self.units, gprice=price)
                    self.wait = wait  # 下一交易发生之前等待的条数
                    self.position = 0
                # 已有空头头寸,然该空头的收益率(价格增长)超过设置的止损率.
                elif self.position == -1 and rc > self.sl:
                    print(50 * "-")
                    if guarantee:
                        price = self.entry_price * (1 + self.sl)
                        print(f"*** STOP LOSS (SHORT | -{self.sl:.4f}) ***")
                    else:
                        print(f"*** STOP LOSS (SHORT | -{rc:.4f}) ***")
                    self.place_buy_order(bar, units=-self.units, gprice=price)
                    self.wait = wait
                    self.position = 0  # 止损单之后,头寸成0;

            # trailing stop loss order
            if self.tsl is not None and self.position != 0:
                # max_price是每次与price比较,并进行更新,总是更新(跟踪)最大值;;
                self.max_price = max(self.max_price, price)
                # min_price是每次与price比较,并进行更新,总是更新(跟踪)最小值;;
                self.min_price = min(self.min_price, price)
                rc_1 = (price - self.max_price) / self.entry_price  # 同最大值比较的收益;
                rc_2 = (self.min_price - price) / self.entry_price  # 同最小值比较的收益;
                if self.position == 1 and rc_1 < -self.tsl:
                    print(50 * "-")
                    print(f"*** TRAILING SL (LONG  | {rc_1:.4f}) ***")
                    self.place_sell_order(bar, units=self.units)
                    self.wait = wait  # 风控事件之后,wait次数恢复;
                    self.position = 0
                elif self.position == -1 and rc_2 < -self.tsl:
                    print(50 * "-")
                    print(f"*** TRAILING SL (SHORT | {rc_2:.4f}) ***")
                    self.place_buy_order(bar, units=-self.units)
                    self.wait = wait
                    self.position = 0

            # take profit order
            if self.tp is not None and self.position != 0:
                rc = (price - self.entry_price) / self.entry_price
                if self.position == 1 and rc > self.tp:
                    print(50 * "-")
                    if guarantee:
                        price = self.entry_price * (1 + self.tp)
                        print(f"*** TAKE PROFIT (LONG  | {self.tp:.4f}) ***")
                    else:
                        print(f"*** TAKE PROFIT (LONG  | {rc:.4f}) ***")
                    self.place_sell_order(bar, units=self.units, gprice=price)
                    self.wait = wait
                    self.position = 0
                elif self.position == -1 and rc < -self.tp:
                    print(50 * "-")
                    if guarantee:
                        price = self.entry_price * (1 - self.tp)
                        print(f"*** TAKE PROFIT (SHORT | {self.tp:.4f}) ***")
                    else:
                        print(f"*** TAKE PROFIT (SHORT | {-rc:.4f}) ***")
                    self.place_buy_order(bar, units=-self.units, gprice=price)
                    self.wait = wait
                    self.position = 0

            action = np.argmax(self.model.predict(state)[0, 0])
            state, reward, done, info = self.env.step(action)
            position = 1 if action == 1 else -1
            # wait初始为5,每个样本,减去1,5此后为0;
            if self.position in [0, -1] and position == 1 and self.wait == 0:
                if self.verbose:
                    print(50 * "-")
                    print(f"{date} | *** GOING LONG ***")
                if self.position == -1:
                    self.place_buy_order(bar - 1, units=-self.units, gprice=None)
                self.place_buy_order(bar - 1, amount=self.current_balance, gprice=None)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1 and self.wait == 0:
                if self.verbose:
                    print(50 * "-")
                    print(f"{date} | *** GOING SHORT ***")
                if self.position == 1:
                    self.place_sell_order(bar - 1, units=self.units, gprice=None)
                self.place_sell_order(bar - 1, amount=self.current_balance, gprice=None)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1

            self.net_wealths.append(
                (
                    date,
                    price,
                    self.units,
                    self.current_balance,
                    self.calculate_net_wealth(price),
                    self.position,
                    self.trades,
                )
            )
        self.net_wealths = pd.DataFrame(
            self.net_wealths,
            columns=[
                "date",
                "price",
                "units",
                "balance",
                "net_wealth",
                "position",
                "trades",
            ],
        )

        self.net_wealths.set_index("date", inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
        self.close_out(bar)
