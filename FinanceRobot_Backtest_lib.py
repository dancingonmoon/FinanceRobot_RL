#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from collections import deque
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, AveragePooling1D
import random
import time

import math
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


def Dataset_Generator(
    data,
    data_features_num,
    date_list,
    lags,
    Batch_size,
    shuffle=False,
    buffer_size=10000,
):
    """
    从交易数据Dataframe,生成dataset, 没有target数据;因为DQN模型的Target为Q_value,是action_Qvalue求得的;
    args:
        data: Dataframe格式的数据(N,features);
        data_features_num:Data的features的数量;
        lags: 样本延时的数量,即dataset window的大小; lags>=1;
        Batch_size:
        shuffle: bool ; 是否shuffle;保持原交易数据的顺序,所以,不能shuffle;缺省为False
    out:
        xs: xs.shape:(date,(N,lags,features));元组,包含date(字符串),以及8个特征的data;date字符串记录交易日的时间戳(例如: "2022-11-19 12:00:00")
    """
    output_signature = tf.TensorSpec((data_features_num), tf.float64)  # data原始为8个特征,8列;
    # gen_f = gen(data)
    xs = tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature,
        args=(data,),
    )  # args用于给gen传递参数,必须是元组的形式传入参数;
    # x_y = tf.data.Dataset.from_tensor_slices(data)

    date = tf.data.Dataset.from_generator(
        gen_date,
        output_signature=tf.TensorSpec((), tf.string),
        # date_list如果是datetime64类型,需要提前转化string: date_list.astrype('string')
        args=(date_list,),
    )  # args用于给gen传递参数,必须是元组的形式传入参数;

    xs = xs.window(lags, shift=1, drop_remainder=True)
    xs = xs.flat_map(lambda w: w.batch(lags, drop_remainder=True))

    date = date.window(lags, shift=1, drop_remainder=True)
    date = date.flat_map(lambda w: w.batch(lags, drop_remainder=True))

    dataset = tf.data.Dataset.zip((date, xs))
    if shuffle == True:
        dataset.shuffle(buffer_size)

    return dataset.batch(Batch_size, drop_remainder=True).prefetch(1)

# 金融沙箱- 模仿OpenAI GYM环境,创建Fiance类, 即,交易市场环境


class observation_space:
    """
    args:
        lags: 样本(交易时刻)延时的个数,表示几个延时交易时刻为一组;
        n_features: 每个交易时刻样本数据的特征数
    """

    def __init__(self, lags, n_features):
        # self.shape = (n,)  # 以后可以改成(lags,features)
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
        return random.randint(0, self.n-1)  # random.randint()是两端都包含的随机值


class Finance_Environment:
    """
    类OpenAI Gym的environment,实现交易state,action,reward,next_state
    """

    def __init__(self, dataset, leverage=1, min_performance=0.85, min_accuracy=0.5,
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
        self.observation_space = observation_space(
            self.lags, self.features)
        self.action_space = action_space(2)  # 假定涨跌两个动作;

    def dataset2data(self, price_Scaler=None, log_return_Scaler=None,
                     price_column=-1, log_return_column=0, Mark_return_column=6):
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
            [], dtype=object)  # object类型,才可以与数值合并到一个array里面;(datetime64类型不行)
        log_return = []
        Mark_return = []
        price = []

        for day, Data in tqdm(self.dataset, total=self.dataset_len,):
            # 1.date:
            day = day[0, -1].numpy()  # 读取date str; lags=-1最后一个时序
            day = np.datetime64(day)  # 字符串转化成datetime64类型;
            date = np.append(date, day)
            # 2. log_return
            logReturn = Data[0, -1, log_return_column]  # lags=-1,log_return第0列
            if log_return_Scaler != None:
                logReturn = log_return_Scaler.inverse_transform(
                    np.array(logReturn).reshape(1, -1))  # 归一化之后的还原;
            log_return = np.append(log_return, logReturn)

            # 3. Mark_return:
            # lags=-1,Mark_return倒数第2列,顺数第6列; 该列归一化前后数值(0或1)无变化
            Mark = Data[0, -1, Mark_return_column]
            Mark_return = np.append(Mark_return, Mark)
            # 4.price:
            price_step = Data[0, -1, price_column]  # lags=-1最后一个时序
            if price_Scaler != None:
                price_step = price_Scaler.inverse_transform(
                    np.array(price_step).reshape(1, -1))  # 归一化之后的还原;
            price = np.append(price, price_step)

            # 合并 date,log_return,Mark_return,price
        env_data = np.c_[date.reshape(-1, 1), log_return.reshape(-1, 1),
                         Mark_return.reshape(-1, 1), price.reshape(-1, 1)]
        return env_data  # (N,4)
        # return date,log_return,Mark_return,price

    def get_state(self, bar):
        # Dataset类型,获得Dataset中,指定序列号的的element, ((N,date),(N,lags,features))
        element = [d for i, d in enumerate(iter(self.dataset)) if i == bar][0]
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

    def step(self,  action):
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
        elif (self.performance < self.min_performance and
              self.bar > 15):  # 当最佳策略reward=0,总收益率低于最小值,步数(时刻)大于(15)次时,结束;即,执行策略action(15)次以后,收益率仍小于最小收益率,action策略仍然是显亏损,就中止结束
            done = True
        elif (self.accuracy < self.min_accuracy and
              self.bar > 15):  # 当执行(15)步以后,产生收益的action的次数,仍低于最小次数时,结束中止,停止训练;
            done = True
        else:
            done = False

        self.state = next(self.iter_dataset)[1]  # 下一个state

        info = {}
        return self.state, reward_1 + reward_2 * 5, done, info  # reward_2*5 不知道为什么放大5倍

# 重写基于Finance类Env的Finance Agent,实现改进了的Agent在模拟的交易环境中,逐次从历史交易数据中学习正确的策略动作,使得action Q值Pari(短期收益+长期收益)总是依最佳策略执行

# 定义一个训练模型,用于DQN强化学习网络中的基础模型,可以替换成其它模型:


def build_model(input_shape, hidden_unit=24, lr=0.001):
    """
    构建模型,model输入为多样本的state的tensor,shape(N,lags,features);此处模型为最简单的3层全连接DNN,可以用其它模型替换.
    """
    model = tf.keras.Sequential()
    model.add(Dense(
        hidden_unit, input_shape=input_shape, activation='relu'))
    # (N,lags,features)->(N,lags,hidden_units)
    model.add(Dense(hidden_unit, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='linear'))  # (N,lags,2)
    model.add(AveragePooling1D(pool_size=input_shape[0],
              strides=1, padding='valid'))  # (N,1,2)
    model.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))
    return model  # output.shape:(N,1,2)

# 基于Finance类Env的Finance Agent:


class FQLAgent():
    """
    learn_env: 装载有训练集数据,模拟训练集数据交易环境;
    valid_env: 装载有验证集数据,模拟验证集数据交易环境;
    build_model: 自建的深度学习模型,在model.compile,或者自定义单步训练后,做变量输入;该模型将在learn_env中训练,同一模型(训练后参数),再到valid_env中验证;
    """

    def __init__(self, build_model, gamma=0.95, , tau=1e-3, learn_env=None, valid_env=None, validation=True, replay_batch_size=2000, target_network_update_freq=2000, fit_batch_size=128):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = gamma
        self.tau = tau

        self.replay_batch_size = replay_batch_size
        self.fit_batch_size = fit_batch_size
        # self.batch_size = batch_size
        self.max_treward = 0
        self.trewards = []
        self.averages = []
        self.performances = []
        self.aperformances = []
        self.vperformances = []
        self.memory = deque(maxlen=2000)
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done'])

        self.Q = build_model  # Q Network model
        self.Q_target = build_model  # Q_target Network model ;同一模型,将有不同的weights;
        self.step_num = 0  # 用于每步训练计数,计数器初始化
        self.loss = []
        self.optimizer = Adam(learning_rate=5e-4)

        self.validation = validation

    def act(self, state):
        """
        对单个state样本,执行behavior_strategy,返回action.
        每个state样本数据.shape(1,lags,features)
        """
        if tf.random.uniform((1,), maxval=1) <= self.epsilon:
            return self.learn_env.action_space.sample()
        actions = self.Q(state)  
        return tf.math.argmax(actions, axis=-1).numpy()[0, 0] # (N,1,2)->(2,) ;

    def softupdate(self, Q, Q_target, tau=1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            Q (tensorflow model): weights will be copied from
            Q_target (Target model): weights will be copied to
            tau (float): interpolation parameter 
        """
        Weights_Q = self.Q.get_weights()
        Weights_Q_target = self.Q_target.get_weights()

        ws_q_target = []
        for w_q, w_q_target in zip(Weights_Q, Weights_Q_target):
            w_q_target = (1-tau) * w_q_target + tau * w_q
            ws_q_target.append(w_q_target)

        Q_target.set_weights(ws_q_target)

        return Q_target

    @tf.function  # 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行。
    def train_step(self, experience_dataset):
        # 求导,根据导数优化变量
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(experience_dataset)
        gradients = tape.gradient(loss_value, self.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.Q.trainable_variables))

        return loss_value

    def loss_func(self, experience_dataset):
        """
        自定义loss函数,
        args:
            state_batch: 一组历史经验中,包含replay_batch_size个state的tensor,shape:(N,lags,tiled_layer,obs_space_n)
            action_batch: shape:(N,)
            reward_batch: shape:(N,)
            next_state_batch: shape:(N,lags,tiled_layer,obs_space_n)
            undone_batch: shape: (N,) bool类型,需要转成float才能参与算术运算;
        """
        state_batch = experience_dataset[0]
        action_batch = experience_dataset[1]
        reward_batch = experience_dataset[2]
        next_state_batch = experience_dataset[3]
        undone_batch = experience_dataset[4]

        Q_next_state = tf.math.reduce_max(
            self.Q_target(next_state_batch), axis=-1)[:, 0]

        TD_Q_target = reward_batch + self.gamma * Q_next_state * undone_batch
        # print('TD_Q_target.shape:{}'.format(TD_Q_target.shape))
        Qvalue_action_pair = self.Q_target(state_batch)  # (N,lags,action_space_n)

        # tf.gather(batch_dims=1)相当于对(N,lags,action_space_n)的首维N进行遍历循环,每个循环tf.gather:(lags,action_space_n)->(lags,);再stack合并成(N,1)
        Q_predict = tf.gather(Qvalue_action_pair,
                              indices=action_batch, axis=-1, batch_dims=1)[:, 0]

        loss_value = tf.keras.losses.mse(TD_Q_target, Q_predict)

        return loss_value

    def replay_train_step(self,):
        """
        使用tensorflow自定义训练的方法,自定义单步训练,loss函数中的y_true,y_predict采用模型输出指定action的Qvalue.
        从memory(历史经验),提取一次batch,学习,更新模型,相当于一个batch的训练
        args:
            callbacks: keras定义的callbacks,用列表装入多个callbacks
        """
        batch = random.sample(
            self.memory, self.replay_batch_size)  # 从memory里面,随机取出repla_batch_size个样本;

        # batch中每个样本,生成,state,action_Qvalue_pair;再组合成dataset,包括(X_dataset,Y_dataset)

        # 星号拆包,送入命名元组,获得field name, (N,lags,field_names)
        batch_Exp = self.experience(*zip(*batch))
        state_batch = tf.convert_to_tensor(
            batch_Exp.state, dtype=tf.float32)  # (N,lags,obs_space_n)
        action_batch = tf.convert_to_tensor(
            batch_Exp.action)  # (N,) 最后一维的值表示action_space的序列

        reward_batch = tf.convert_to_tensor(batch_Exp.reward, dtype=tf.float32)  # (N,)
        next_state_batch = tf.convert_to_tensor(
            batch_Exp.next_state, dtype=tf.float32)  # (N,lags,obs_space_n)
        undone_batch = tf.logical_not(batch_Exp.done)  # (N,)
        # undone_batch原为(N,)bool类型,需要转成float才能参与算术运算
        undone_batch = tf.cast(undone_batch, dtype=tf.float32)

        state_batch = tf.data.Dataset.from_tensor_slices(state_batch)
        action_batch = tf.data.Dataset.from_tensor_slices(action_batch)
        reward_batch = tf.data.Dataset.from_tensor_slices(reward_batch)
        next_state_batch = tf.data.Dataset.from_tensor_slices(next_state_batch)
        undone_batch = tf.data.Dataset.from_tensor_slices(undone_batch)

        experience_dataset = tf.data.Dataset.zip((state_batch, action_batch, reward_batch, next_state_batch, undone_batch)).batch(
            self.fit_batch_size).prefetch(1)

        # 训练一次,优化一次weights:
        for experience_dataset_batch in experience_dataset:
            train_loss_avg = tf.keras.metrics.Mean()  # metrics类初始化
            # 并不是模型的输出计算loss,而是loss指定action的Qvalue_action_pair
            loss_value = self.train_step(experience_dataset_batch)
            # Track progress
            train_loss_avg.update_state(loss_value)  # Add current batch loss

        # 采用soft update的方法,soft update Q target Network:
        self.softupdate(self.Q, self.Q_target, self.tau)

        self.loss.append(train_loss_avg.result())
        print('step_num: {} | loss: {:.4f} '.format(
            self.step_num, train_loss_avg.result()))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, callbacks=None):
        """
        从memory(历史经验),提取一次batch,学习,更新模型,相当于一个batch的训练
        args:
            callbacks: keras定义的callbacks,用列表装入多个callbacks
        """
        batch = random.sample(
            self.memory, self.replay_batch_size)  # 从memory里面,随机取出repla_batch_size个样本;

        # batch中每个样本,生成,state,action_Qvalue_pair;再组合成dataset,包括(X_dataset,Y_dataset)

        # 星号拆包,送入命名元组,获得field name, (N,lags,field_names)
        batch_Exp = self.experience(*zip(*batch))
        state_batch = tf.convert_to_tensor(
            batch_Exp.state, dtype=tf.float32)  # (N,lags,obs_space_n)
        action_batch = tf.convert_to_tensor(
            batch_Exp.action)  # (N,) 最后一维的值表示action_space的序列

        reward_batch = tf.convert_to_tensor(batch_Exp.reward, dtype=tf.float32)  # (N,)
        next_state_batch = tf.convert_to_tensor(
            batch_Exp.next_state, dtype=tf.float32)  # (N,lags,obs_space_n)
        undone_batch = tf.logical_not(batch_Exp.done)  # (N,)
        # undone_batch原为(N,)bool类型,需要转成float才能参与算术运算
        undone_batch = tf.cast(undone_batch, dtype=tf.float32)

        # (N,lags,action_space_n) -> (N,lags) ->(N,)
        Q_next_state = tf.math.reduce_max(
            self.Q_target(next_state_batch), axis=-1)[:, 0]
        # print('state_batch.shape:{}'.format(state_batch.shape))
        # print('next_state_batch.shape:{}'.format(next_state_batch.shape))
        # print('Q_target(next_state_batch).shape:{}'.format(self.Q_target(next_state_batch).shape))
        # print('max(Q_target(next_state_batch)).shape:{}'.format(tf.math.reduce_max(self.Q_target(next_state_batch),axis=-1).shape))
        # print('Q_next_state.shape:{}'.format(Q_next_state.shape))
        # print('undone_batch.shape:{}'.format(undone_batch.shape))
        # print('reward_batch.shape:{}'.format(reward_batch.shape))

        # (N,);都是(N,)的矩阵算术运算.
        TD_Q_target = reward_batch + self.gamma * Q_next_state * undone_batch
        # print('TD_Q_target.shape:{}'.format(TD_Q_target.shape))
        # TD_Q_target = tf.expand_dims(TD_Q_target,axis=1) #(N,) -> (N,lags=1)
        Qvalue_action_pair = self.Q_target(state_batch)
        Qvalue_action_pair = Qvalue_action_pair.numpy()  # numpy可以item assignment,而tensor不可以
        # Qvalue_action_pair = self.Q_target.predict(
        #     state_batch, verbose=0)  # (N,lags,action_space_n) predict输出的是numpy
        # 以下尝试使用tf.gather方法失败,因为Qvalue_action_pair需要对其中的某些元素赋值,其它值不变化.tf.gather是挑选出条件值
        # # tf.gather(batch_dims=1)相当于对(N,lags,action_space_n)的首维N进行遍历循环,每个循环tf.gather:(lags,action_space_n)->(lags,1);再stack合并成(N,lags,1)
        # Qvalue_action_pair = tf.gather(Qvalue_action_pair,indices=action_batch,batch_dims=1)
        # Qvalue_action_pair = TD_Q_target #更新
        # tf.vectorized_map(lambda i,a: Qvalue_action_pair[i,:,a]=TD_Q_target[i], enumerate(action_batch))
        # 或者采用遍历循环的方法,仍然未逃离for循环 :
        for i, a in enumerate(action_batch):
            Qvalue_action_pair[i, :, a] = TD_Q_target[i]

        # print('Qvalue_action_pair.shape:{}'.format(Qvalue_action_pair.shape))
        X_dataset = tf.data.Dataset.from_tensor_slices(state_batch)
        y_dataset = tf.data.Dataset.from_tensor_slices(Qvalue_action_pair)
        Xy_dataset = tf.data.Dataset.zip((X_dataset, y_dataset)).batch(
            self.fit_batch_size).prefetch(1)
        # epochs = int(self.replay_batch_size/self.fit_batch_size) + 1

        # Dataset,训练(从历史经验中学习);epochs=1 而不是replay_batch_size/fit_batch_size,因为每一个epoch都是所有dateset全部训练一次
        history = self.Q.fit(Xy_dataset, epochs=1,
                             callbacks=callbacks,
                             verbose=False)  # 获得一个dataset样本的更新,立即训练模型,更新模型 verbose=0关闭每个样本进度条

        # 采用soft update的方法,soft update Q target Network:
        self.softupdate(self.Q, self.Q_target, self.tau)

        self.loss.append(history.history['loss'])
        print('step_num: {} | loss: {:.4f} '.format(
            self.step_num, history.history['loss'][0]))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes, callbacks):
        """
        args:
            episodes: 训练轮数;
            callbacks: keras定义的callbacks,用列表装入多个callbacks.
        """
        # 游戏回合开始: env跑一遍;每跑一遍,生成一次样本数据,送入Memory,再学习一次;共学习episodes次
        start_time = time.time()
        total_time = time.time()
        for episode in range(episodes):
            state = self.learn_env.reset()  # reset输出为(N,lags,features)
            # print('e:{};次reset之后的self.bar:{}'.format(episode, self.learn_env.bar))
            # print('state:{}'.format(state))
            # state = tf.expand_dims(state, 0) #已经(N,lags,features),不再扩展
            for _ in range(10000):  # 大于训练集的最大长度的值,让每个样本参与训练
                action = self.act(state)

                next_state, reward, done, info = self.learn_env.step(
                    action)
                # print('e:{},step:{}之后的self.bar:{}'.format(
                #     episode, _, self.learn_env.bar))
                # print('learn_env.next_state:{}'.format(next_state))
                # next_state = tf.expand_dims(next_state, 0)
                experience = self.experience(
                    state, action, reward, next_state, done)
                self.memory.append(experience)
                # self.memory.append(
                #     [state, action, reward, next_state, done])  # 注:最后一条样本因为没有next(),在env.step方法输出的state是原state
                # print('memory.len:{}, memory[-1]:{}'.format(len(self.memory),self.memory[-1]))
                # print('memory.len:{},done:{}'.format(len(self.memory), done))
                state = next_state
                if done:  # 当游戏结束,或者意外中止时,记录,再进入下一个回合:
                    treward = _ + 1
                    self.trewards.append(treward)
                    # 这里不可以取平均值吗? np.average()
                    average = np.mean(self.trewards[-25:])
                    profit_rate = self.learn_env.performance
                    self.averages.append(average)
                    self.performances.append(profit_rate)
                    self.aperformances.append(np.mean(self.performances[-25:]))

                    self.max_treward = max(self.max_treward, treward)
                    time_assumed = (time.time() - start_time)/60
                    total_time_assumed = (time.time()-total_time)/60
                    text = 'episode:{:4d}/{},耗时:{:3.2f}分/{:3.2f},训练集: | average_treward: {:6.1f} | max_treward: {:4d} | profit_rate: {:5.3f} '
                    # \r 默认表示将输出的内容返回到第一个指针，这样的话，后面的内容会覆盖前面的内容
                    print(text.format(episode+1,  episodes, time_assumed,
                          total_time_assumed, average, self.max_treward, profit_rate))
                    start_time = time.time()

                    break
            if self.validation:
                self.validate(episode, episodes)

            # 开始模型更新,即从样本数据中学习一次:
            if len(self.memory) > self.replay_batch_size:  # memory的样本数超过batch,即开始从历史经验中学习
                self.replay(callbacks)

    def validate(self, episode, episodes):

        state = self.valid_env.reset()
        # state = tf.expand_dims(state, 0)

        for i in range(10000):
            # learn_env.act(state)有根据epsilon做随机,验证集还是直接用model输出Q值,再argmax选动作
            action = tf.math.argmax(self.Q(
                state), axis=-1).numpy()[0, 0]
            next_state, reward, done, info = self.valid_env.step(
                action)
            state = next_state
            if done:
                treward = i + 1
                profit_rate = self.valid_env.performance
                self.vperformances.append(profit_rate)
                if (episode+1) % 10 == 0:  # 每10回合,即10个batch_size
                    text = 71 * '-'
                    text += '\nepisode:{:4d}/{},验证集: | treward: {:4d} | profit_rate: {:5.3f} |'
                    print(text.format(episode+1,  episodes, treward, profit_rate))

                break

# agent的向量化backtesting,注意的是:
# 其就训练后的模型,做出了每个样本的预测,根据预测后的action_Qvalue队,预测了策略action,根据action,增加了一列头寸


def Backtesting_vector(agent_model, env, price_Scaler=None, log_return_Scaler=None,
                       price_column=-1, log_return_column=0, Mark_return_column=6):
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

    env_data = env.dataset2data(price_Scaler=price_Scaler, log_return_Scaler=log_return_Scaler,
                                price_column=price_column,
                                log_return_column=log_return_column,
                                Mark_return_column=Mark_return_column)

    done = False
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
    strategy_return = positions * env.leverage * \
        env_data[:, 1]  # 没有乘以价格基数,日收益率如何反映真实的收益?因为时刻连续,时刻累计
    env_data = np.c_[env_data, strategy_return, positions]  # (N,6)
    return env_data

# 基于事件的回测 Event_based Backtest


class Backtesting_event:
    """
    Event based Backtesting
    """

    def __init__(self, env, model, amount, percent_commission, fixed_commission, verbose=False,
                 price_Scaler=None, log_return_Scaler=None, MinUnit_1Position=0):
        ''' 
        args:
            env: 类OpenAI GYM的Finance 类环境,由Fiance_environment类生成;给定state,以及action,能够生成next_state,reward;
            model: 训练后的DQN模型,能够针对state给出策略action;
            amount: 回测的初始金额;
            percent_commission: transaction时的与价格相比例的交易手续费;
            fixed_commission: transaction时,一次性,固定的交易手续费
            price_Scaler: 收盘价归一化的Scaler;
            log_return_Scaler: log_return归一化的Scaler
            MinUnit_1Position:头寸的最小计量单位;对于股票而言,最小的单位是整数,即为10的0次方,取值0;对于BTC而言,最小的单位是10的-8次方,即取-8;如果某只股票最小单位为10,则值为1;
        '''
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
            price_Scaler=price_Scaler, log_return_Scaler=log_return_Scaler)
        self.MinUnit_1Position = MinUnit_1Position

    def get_date_price(self, bar):
        ''' Returns date and price for a given bar.
        '''
        date = self.env_data[bar, 0]
        price = self.env_data[bar, -1]
        return date, price

    def print_balance(self, bar):
        ''' Prints the current cash balance for a given bar.
        '''
        date, price = self.get_date_price(bar)
        print(f'{date} | current balance = {self.current_balance:.2f}')

    def calculate_net_wealth(self, price):
        return self.current_balance + self.units * price

    def print_net_wealth(self, bar):
        ''' Prints the net wealth for a given bar
            (cash + position).
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        # print(f'{date} | net wealth = {net_wealth:.2f}')
        print('{} | balance:{:.2f}+units:{}*price:{:.4f}=net_wealth:{:.4f}'.format(date,
              self.current_balance, self.units, price, net_wealth))

    def set_prices(self, price):
        ''' Sets prices for tracking of performance.
            To test for e.g. trailing stop loss hit.
        '''
        self.entry_price = price
        self.min_price = price
        self.max_price = price

    def place_buy_order(self, bar, amount=None, units=None, gprice=None):
        ''' Places a buy order for a given bar and for
            a given amount or number of units. 当units<0,表示买空;
        args:
            gprice: 买单的指定价格,guarantee价格;
        '''
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            MinUnit_1Position = 10 ** self.MinUnit_1Position
            units = int(amount / price / MinUnit_1Position) * \
                MinUnit_1Position  # 获得低于指定位数小数的值;
            # print('units({})=int(amount({})/price({}))'.format(units, amount, price))
            # units = amount / price  # alternative handling
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc
        self.units += units
        self.trades += 1
        self.set_prices(price)
        if self.verbose:
            # print(f'{date} | buy {units} units for {price:.4f}')
            print("{}\'s price {:0.4f}, buy {} units.".format(date, price, units,))
            self.print_balance(bar)

    def place_sell_order(self, bar, amount=None, units=None, gprice=None):
        ''' Places a sell order for a given bar and for
            a given amount or number of units.
        args:
            gprice: 卖单的指定价格,guarantee价格;
        '''
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            MinUnit_1Position = 10 ** self.MinUnit_1Position
            units = int(amount / price / MinUnit_1Position) * \
                MinUnit_1Position  # 获得低于指定位数小数的值;
            # units = amount / price  # altermative handling
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        self.trades += 1
        self.set_prices(price)
        if self.verbose:
            print("{}\'s price {:0.4f}, sell {} units.".format(date, price, units))
            self.print_balance(bar)

    def close_out(self, bar):
        ''' Closes out any open position at a given bar.
        '''
        date, price = self.get_date_price(bar)
        print(50 * '=')
        print(f'{date} | *** CLOSING OUT ***')
        if self.units < 0:
            self.place_buy_order(bar, units=-self.units)
        else:
            self.place_sell_order(bar, units=self.units)
        if not self.verbose:
            print(f'{date} | current balance = {self.current_balance:.2f}')
        self.net_performance = (self.current_balance / self.initial_amount - 1) * 100
        print(f'{date} | net performance [%] = {self.net_performance:.4f}')
        print(f'{date} | number of trades [#] = {self.trades}')
        print(50 * '=')

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
        action = np.argmax(self.model.predict(
            state)[0, 0])
        state, reward, done, info = self.env.step(action)  # 因为bar从1开始,state从0跳到1
        # bar不是从0开始,而是从1开始,是因为,初始第一条是第0时刻的当日收盘价,没有上个收盘价,无法下单;下单应从第1个时刻开始;
        for bar in range(1, self.env_data.shape[0]):
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * '=')
                print(f'{date} | *** START BACKTEST ***')
                self.print_balance(bar)
                print(50 * '=')
            action = np.argmax(self.model.predict(
                state)[0, 0])
            state, reward, done, info = self.env.step(action)
            position = 1 if action == 1 else -1
            if self.position in [0, -1] and position == 1:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING LONG ***')
                if self.position == -1:  # 买单的价格是前一交易日的收盘价,所以bar-1,意味着买单的价格; #self.position=-1,表明手中有空头,先买空头,再买多头
                    self.place_buy_order(
                        bar - 1, units=-self.units, gprice=None)
                self.place_buy_order(bar - 1,
                                     amount=self.current_balance, gprice=None)  # 先买空头,再买多头;
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING SHORT ***')
                if self.position == 1:
                    # 为什么是bar-1,前一日呢? 因为,place_sell_order方法是以指定样本的价格来买,该样本的价格是收盘价,所以是上一个样本的时间
                    self.place_sell_order(bar - 1, units=self.units, gprice=None)
                self.place_sell_order(bar - 1,
                                      amount=self.current_balance, gprice=None)  # 手中已经卖出全部头寸,只有现金,再卖空(先借头寸,再卖出空头,卖出空头,头寸的余额为负值)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1
            # 以下输出资产/交易状态:

            self.net_wealths.append((date, price, self.units, self.current_balance,
                                     self.calculate_net_wealth(price), self.position, self.trades))
        self.net_wealths = pd.DataFrame(self.net_wealths,
                                        columns=['date', 'price', 'units', 'balance', 'net_wealth', 'position', 'trades'])
        self.net_wealths.set_index('date', inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(
            self.net_wealths.index)
        self.close_out(bar)

    def backtest_strategy_WH_RM(self, StopLoss=None, TrailStopLoss=None, TakeProfit=None,
                                wait=5, guarantee=False):
        ''' Event-based backtesting of the trading bot's performance.
            Incl. stop loss, trailing stop loss and take profit.
            利用BacktesingBase类中定义的交易的方法,买,卖等;实现观察数据的回测过程.
            带有risk Management;包含:止损,跟踪止损,止盈等风控措施;
        args:
            StopLoss: stop loss,止损;
            TrailStopLoss: trailing stop loss,跟踪止损;
            TakeProfit: take profit,止盈;
            wait: 两次策略交易事件(风控事件,或者买卖交易)之间等待的条数(样本条数,或者是样本间隔交易间隔的数量)
            guarantee: bool值,表示是否以保证价格,或是市场当时的价格成交;
        '''
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
        action = np.argmax(self.model.predict(
            state)[0, 0])
        state, reward, done, info = self.env.step(action)  # bar从1开始,跳到第1个样本;
        for bar in range(1, self.env_data.shape[0]):
            self.wait = max(0, self.wait - 1)
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * '=')
                print(f'{date} | *** START BACKTEST ***')
                self.print_balance(bar)
                print(50 * '=')

            # stop loss order
            if self.sl is not None and self.position != 0:  # 定义了止损,并且已有头寸,无论空,还是多
                # 根据最后一笔交易的进入价格(持有头寸的买卖价格),计算收益
                rc = (price - self.entry_price) / self.entry_price
                # 已有多头头寸,在此交易日(bar),该头寸的持有收益亏损超过设置的止损率.(1->2倍的ATR)
                if self.position == 1 and rc < -self.sl:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 - self.sl)  # 成交价格设置成指定的止损价格;
                        print(f'*** STOP LOSS (LONG  | {-self.sl:.4f}) ***')
                    else:  # 否则,就是交易日的当时价格成交; 当时的价格可能时动态的,这里采用当日的收盘价成交;亦即,假设在当日收盘价之后,下一日到来之前交易;
                        print(f'*** STOP LOSS (LONG  | {rc:.4f}) ***')
                    self.place_sell_order(bar, units=self.units, gprice=price)
                    self.wait = wait  # 下一交易发生之前等待的条数
                    self.position = 0
                # 已有空头头寸,然该空头的收益率(价格增长)超过设置的止损率.
                elif self.position == -1 and rc > self.sl:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 + self.sl)
                        print(f'*** STOP LOSS (SHORT | -{self.sl:.4f}) ***')
                    else:
                        print(f'*** STOP LOSS (SHORT | -{rc:.4f}) ***')
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
                    print(50 * '-')
                    print(f'*** TRAILING SL (LONG  | {rc_1:.4f}) ***')
                    self.place_sell_order(bar, units=self.units)
                    self.wait = wait  # 风控事件之后,wait次数恢复;
                    self.position = 0
                elif self.position == -1 and rc_2 < -self.tsl:
                    print(50 * '-')
                    print(f'*** TRAILING SL (SHORT | {rc_2:.4f}) ***')
                    self.place_buy_order(bar, units=-self.units)
                    self.wait = wait
                    self.position = 0

            # take profit order
            if self.tp is not None and self.position != 0:
                rc = (price - self.entry_price) / self.entry_price
                if self.position == 1 and rc > self.tp:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 + self.tp)
                        print(f'*** TAKE PROFIT (LONG  | {self.tp:.4f}) ***')
                    else:
                        print(f'*** TAKE PROFIT (LONG  | {rc:.4f}) ***')
                    self.place_sell_order(bar, units=self.units, gprice=price)
                    self.wait = wait
                    self.position = 0
                elif self.position == -1 and rc < -self.tp:
                    print(50 * '-')
                    if guarantee:
                        price = self.entry_price * (1 - self.tp)
                        print(f'*** TAKE PROFIT (SHORT | {self.tp:.4f}) ***')
                    else:
                        print(f'*** TAKE PROFIT (SHORT | {-rc:.4f}) ***')
                    self.place_buy_order(bar, units=-self.units, gprice=price)
                    self.wait = wait
                    self.position = 0

            action = np.argmax(self.model.predict(state)[0, 0])
            state, reward, done, info = self.env.step(action)
            position = 1 if action == 1 else -1
            # wait初始为5,每个样本,减去1,5此后为0;
            if self.position in [0, -1] and position == 1 and self.wait == 0:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING LONG ***')
                if self.position == -1:
                    self.place_buy_order(bar-1, units=-self.units, gprice=None)
                self.place_buy_order(bar-1, amount=self.current_balance, gprice=None)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1 and self.wait == 0:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING SHORT ***')
                if self.position == 1:
                    self.place_sell_order(bar-1, units=self.units, gprice=None)
                self.place_sell_order(bar-1, amount=self.current_balance, gprice=None)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1

            self.net_wealths.append((date, price, self.units, self.current_balance,
                                     self.calculate_net_wealth(price), self.position, self.trades))
        self.net_wealths = pd.DataFrame(self.net_wealths,
                                        columns=['date', 'price', 'units', 'balance', 'net_wealth', 'position', 'trades'])

        self.net_wealths.set_index('date', inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(self.net_wealths.index)
        self.close_out(bar)


# 自建线性模型二,受论文(AreTransformerReallyMatter)启发而就;
# 1. 时间序列拆解: seasonal, trend
class series_decomp(tf.keras.layers.Layer):
    """
    Series decomposition block
    """

    def __init__(self, pool_size, name="series_comp", **kwargs):
        """
        input:
            pool_size: AveragePooling1D时,求平均值的窗口大小;如果pool_size-1为奇数,右边比左边多填1组;pool_size-1为偶数,则左右两边填充相等的组数;故而,pool_size设为奇数最佳
        output:
            res: sesonal component, shape与input一致
            moving_eman: trend cyclical component,shape与input一致
        """
        super(series_decomp, self).__init__(name=name, **kwargs)
        self.moving_avg = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size, strides=1, padding="same"
        )  # 'same'时,经过查API文档,不是通过填充来补齐shape,而是求平均值的时候,求不同shape的平均值,即不足shape,平均值是剩余shape的平均值;所以,不需要补填充

    def call(self, x):
        moving_mean = self.moving_avg(x)  # shape remains unchanged
        res = x - moving_mean  # shape remains unchanged
        # print(moving_mean.shape)
        return res, moving_mean

# 2. 结合分解后的seasonal/trend,进入Feed Forward前向反馈网络,再输出到指定的形状:


class Decompose_FF_Linear(tf.keras.Model):
    """
    Decomposition-Feed Forward-Liner,该模型用于DQN网络中,基础模型,预测股市买卖动作;
    """

    def __init__(self, seq_len, in_features, out_features,
                 kernel_size=25, dropout=0.3, name="Decompose_FF_Linear", **kwargs):
        """
        seq_len: 输入序列长度;
        in_features: 输入预测特征数;
        pred_len: 输出序列长度
        out_features: 输出序列的特征数;
        kernel_size: moving_avg(即:AveragePooling1D)时的pool_size(窗口大小)
        """
        super(Decompose_FF_Linear, self).__init__(name=name, **kwargs)
        self.seq_len = seq_len
        # self.pred_len = pred_len
        self.in_features = in_features
        self.out_features = out_features

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        # Feed Forward
        FF_hidden = 4 * in_features
        self.FF_Seasonal_Dense0 = tf.keras.layers.Dense(FF_hidden, activation='relu')
        self.FF_Seasonal_Dense1 = tf.keras.layers.Dense(in_features, activation='relu')
        self.FF_dropout = tf.keras.layers.Dropout(dropout)
        # Conv1D:
        self.Conv1D_Seasonal = tf.keras.layers.Conv1D(
            filters=out_features, kernel_size=seq_len, strides=1, padding='valid', activation=None)
        self.Conv1D_Trend = tf.keras.layers.Conv1D(
            filters=out_features, kernel_size=seq_len, strides=1, padding='valid', activation=None)

    def call(self, x):
        # x: [Batch, seq_len, in_features]
        seasonal_init, trend_init = self.decompsition(x)  # (Batch,seq_len,in_features)

        # Feed Forward:
        seasonal_x = self.FF_Seasonal_Dense0(
            seasonal_init)  # (Batch,seq_len,4*in_features)
        seasonal_x = self.FF_dropout(seasonal_x)
        seasonal_x = self.FF_Seasonal_Dense1(seasonal_x)  # (Batch,seq_len,in_features)
        seasonal_x = self.FF_dropout(seasonal_x)
        seasonal_x += seasonal_init

        # Conv1D:
        seasonal_x = self.Conv1D_Seasonal(seasonal_x)  # (Batch,1,out_features)
        trend_x = self.Conv1D_Trend(trend_init)  # (Batch,1,out_features)

        # 合并:
        x = seasonal_x + trend_x  # (Batch,1,out_features)

        return x
