import tensorflow as tf

from collections import deque, namedtuple
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Dropout, AveragePooling1D
import random
import time

# from tqdm import tqdm

import numpy as np


# import pandas as pd


# 定义一个训练模型,用于DQN强化学习网络中的基础模型,可以替换成其它模型:
def build_model(input_shape, hidden_unit=24, lr=0.001):
    """
    构建模型,model输入为多样本的state的tensor,shape(N,lags,features);此处模型为最简单的3层全连接DNN,可以用其它模型替换.
    """
    model = tf.keras.Sequential()
    model.add(Dense(hidden_unit, input_shape=input_shape, activation="relu"))
    # (N,lags,features)->(N,lags,hidden_units)
    model.add(Dense(hidden_unit, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="linear"))  # (N,lags,2)
    model.add(
        AveragePooling1D(pool_size=input_shape[0], strides=1, padding="valid")
    )  # (N,1,2)
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=lr))
    return model  # output.shape:(N,1,2)


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

    def __init__(
            self,
            seq_len,
            in_features,
            out_features,
            kernel_size=25,
            dropout=0.3,
            name="Decompose_FF_Linear",
            **kwargs,
    ):
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
        FF_hidden0 = 32 * in_features
        self.FF_Seasonal_Dense0 = tf.keras.layers.Dense(FF_hidden0, activation="relu",
                                                        kernel_initializer=tf.keras.initializers.Orthogonal())
        self.FF_Seasonal_Dense1 = tf.keras.layers.Dense(in_features, activation="relu",
                                                        kernel_initializer=tf.keras.initializers.Orthogonal())
        self.FF_dropout = tf.keras.layers.Dropout(dropout)
        # Conv1D:
        self.Conv1D_Seasonal = tf.keras.layers.Conv1D(
            filters=out_features,
            kernel_size=seq_len,
            strides=1,
            padding="valid",
            activation=None,
        )
        self.Conv1D_Trend = tf.keras.layers.Conv1D(
            filters=out_features,
            kernel_size=seq_len,
            strides=1,
            padding="valid",
            activation=None,
        )

    def call(self, x):
        # x: [Batch, seq_len, in_features]
        seasonal_init, trend_init = self.decompsition(x)  # (Batch,seq_len,in_features)

        # Feed Forward:
        seasonal_x = self.FF_Seasonal_Dense0(seasonal_init)  # (Batch,seq_len,32*in_features)
        seasonal_x = self.FF_dropout(seasonal_x)
        seasonal_x = self.FF_Seasonal_Dense1(seasonal_x)  # (Batch,seq_len,in_features)
        seasonal_x = self.FF_dropout(seasonal_x)
        seasonal_x += seasonal_init

        # Conv1D:
        seasonal_x = self.Conv1D_Seasonal(seasonal_x)  # (Batch,1,out_features)
        trend_x = self.Conv1D_Trend(trend_init)  # (Batch,1,out_features)

        # 合并:
        x = seasonal_x + trend_x  # (Batch,1,out_features)
        # x = tf.squeeze(x, axis=1)  # (Batch,out_features)

        return x


# 基于Finance类Env的Finance Agent:


class FQLAgent_DQN:
    """
    learn_env: 装载有训练集数据,模拟训练集数据交易环境;
    valid_env: 装载有验证集数据,模拟验证集数据交易环境;
    build_model: 自建的深度学习模型,在model.compile,或者自定义单步训练后,做变量输入;该模型将在learn_env中训练,同一模型(训练后参数),再到valid_env中验证;
    """

    def __init__(
            self,
            build_model,
            learning_rate=5e-4,
            gamma=0.95,
            tau=1e-3,
            learn_env=None,
            valid_env=None,
            validation=True,
            memory_size=10000,
            replay_batch_size=2000,
            target_network_update_freq=2000,
            fit_batch_size=128,
    ):
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
        self.trewards = deque(maxlen=50)
        self.averages = []
        self.performances = []
        self.aperformances = []
        self.vperformances = []
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"]
        )

        self.Q = build_model  # Q Network model
        self.Q_target = build_model  # Q_target Network model ;同一模型,将有不同的weights;
        self.step_num = 0  # 用于每步训练计数,计数器初始化
        self.loss = []
        self.optimizer = Adam(learning_rate=learning_rate, )

        self.validation = validation

    def act(self, state):
        """
        对单个state样本,执行behavior_strategy,返回action.
        每个state样本数据.shape(1,lags,features)
        """
        if tf.random.uniform((1,), maxval=1) <= self.epsilon:
            return self.learn_env.action_space.sample()
        actions = self.Q(state)
        return tf.math.argmax(actions, axis=-1).numpy()[0, 0]  # (N,1,2)->(2,) ;

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
            w_q_target = (1 - tau) * w_q_target + tau * w_q
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

        Q_next_state = tf.math.reduce_max(self.Q_target(next_state_batch), axis=-1)[
                       :, 0
                       ]

        TD_Q_target = reward_batch + self.gamma * Q_next_state * undone_batch
        # print('TD_Q_target.shape:{}'.format(TD_Q_target.shape))
        Qvalue_action_pair = self.Q_target(state_batch)  # (N,lags,action_space_n)

        # tf.gather(batch_dims=1)相当于对(N,lags,action_space_n)的首维N进行遍历循环,每个循环tf.gather:(lags,action_space_n)->(lags,);再stack合并成(N,1)
        Q_predict = tf.gather(
            Qvalue_action_pair, indices=action_batch, axis=-1, batch_dims=1
        )[:, 0]

        loss_value = tf.keras.losses.mse(TD_Q_target, Q_predict)

        return loss_value

    def replay_train_step(
            self,
    ):
        """
        使用tensorflow自定义训练的方法,自定义单步训练,loss函数中的y_true,y_predict采用模型输出指定action的Qvalue.
        从memory(历史经验),提取一次batch,学习,更新模型,相当于一个batch的训练
        args:
            callbacks: keras定义的callbacks,用列表装入多个callbacks
        """
        batch = random.sample(
            self.memory, self.replay_batch_size
        )  # 从memory里面,随机取出repla_batch_size个样本;

        # batch中每个样本,生成,state,action_Qvalue_pair;再组合成dataset,包括(X_dataset,Y_dataset)

        # 星号拆包,送入命名元组,获得field name, (N,lags,field_names)
        batch_Exp = self.experience(*zip(*batch))
        state_batch = tf.convert_to_tensor(
            batch_Exp.state, dtype=tf.float32
        )  # (1,lags,obs_space_n) ->(lags,obs_space_n),后面转成dataset,增加一维度
        action_batch = tf.convert_to_tensor(
            batch_Exp.action
        )  # (N,) 最后一维的值表示action_space的序列

        reward_batch = tf.convert_to_tensor(batch_Exp.reward, dtype=tf.float32)  # (N,)
        next_state_batch = tf.convert_to_tensor(
            batch_Exp.next_state, dtype=tf.float32
        )  # (N,lags,obs_space_n)
        undone_batch = tf.logical_not(batch_Exp.done)  # (N,)
        # undone_batch原为(N,)bool类型,需要转成float才能参与算术运算
        undone_batch = tf.cast(undone_batch, dtype=tf.float32)

        state_batch = tf.data.Dataset.from_tensor_slices(state_batch)
        action_batch = tf.data.Dataset.from_tensor_slices(action_batch)
        reward_batch = tf.data.Dataset.from_tensor_slices(reward_batch)
        next_state_batch = tf.data.Dataset.from_tensor_slices(next_state_batch)
        undone_batch = tf.data.Dataset.from_tensor_slices(undone_batch)

        experience_dataset = (
            tf.data.Dataset.zip(
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    undone_batch,
                )
            )
            .batch(self.fit_batch_size)
            .prefetch(1)
        )

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
        print(
            "step_num: {} | loss: {:.4f} ".format(
                self.step_num, train_loss_avg.result()
            )
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, callbacks=None):
        """
        从memory(历史经验),提取一次batch,学习,更新模型,相当于一个batch的训练
        args:
            callbacks: keras定义的callbacks,用列表装入多个callbacks
        """
        batch = random.sample(
            self.memory, self.replay_batch_size
        )  # 从memory里面,随机取出repla_batch_size个样本;

        # batch中每个样本,生成,state,action_Qvalue_pair;再组合成dataset,包括(X_dataset,Y_dataset)

        # 星号拆包,送入命名元组,获得field name, (N,lags,field_names)
        batch_Exp = self.experience(*zip(*batch))
        state_batch = tf.convert_to_tensor(
            batch_Exp.state, dtype=tf.float32
        )  # (N,lags,obs_space_n)
        print("state_batch.shape={}".format(state_batch.shape))
        action_batch = tf.convert_to_tensor(
            batch_Exp.action
        )  # (N,) 最后一维的值表示action_space的序列

        reward_batch = tf.convert_to_tensor(batch_Exp.reward, dtype=tf.float32)  # (N,)
        next_state_batch = tf.convert_to_tensor(
            batch_Exp.next_state, dtype=tf.float32
        )  # (N,lags,obs_space_n)
        undone_batch = tf.logical_not(batch_Exp.done)  # (N,)
        # undone_batch原为(N,)bool类型,需要转成float才能参与算术运算
        undone_batch = tf.cast(undone_batch, dtype=tf.float32)

        # (N,lags,action_space_n) -> (N,lags) ->(N,)
        Q_next_state = tf.math.reduce_max(self.Q_target(next_state_batch), axis=-1)[
                       :, 0
                       ]
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
        Qvalue_action_pair = (
            Qvalue_action_pair.numpy()
        )  # numpy可以item assignment,而tensor不可以

        # 或者采用遍历循环的方法,仍然未逃离for循环 :
        for i, a in enumerate(action_batch):
            Qvalue_action_pair[i, :, a] = TD_Q_target[i]

        # print('Qvalue_action_pair.shape:{}'.format(Qvalue_action_pair.shape))
        X_dataset = tf.data.Dataset.from_tensor_slices(state_batch)
        y_dataset = tf.data.Dataset.from_tensor_slices(Qvalue_action_pair)
        Xy_dataset = (
            tf.data.Dataset.zip((X_dataset, y_dataset))
            .batch(self.fit_batch_size)
            .prefetch(1)
        )
        # epochs = int(self.replay_batch_size/self.fit_batch_size) + 1

        # Dataset,训练(从历史经验中学习);epochs=1 而不是replay_batch_size/fit_batch_size,因为每一个epoch都是所有dateset全部训练一次
        history = self.Q.fit(
            Xy_dataset, epochs=1, callbacks=callbacks, verbose=False
        )  # 获得一个dataset样本的更新,立即训练模型,更新模型 verbose=0关闭每个样本进度条

        # 采用soft update的方法,soft update Q target Network:
        self.softupdate(self.Q, self.Q_target, self.tau)

        self.loss.append(history.history["loss"])
        print(
            "step_num: {} | loss: {:.4f} ".format(
                self.step_num, history.history["loss"][0]
            )
        )

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
            # print("state.shape:{}".format(state.shape))
            # state = tf.expand_dims(state, 0) #已经(N,lags,features),不再扩展
            for _ in range(10000):  # 大于训练集的最大长度的值,让每个样本参与训练
                action = self.act(state)

                next_state, reward, done, info = self.learn_env.step(action)
                # print('e:{},step:{}之后的self.bar:{}'.format(
                #     episode, _, self.learn_env.bar))
                # print('learn_env.next_state:{}'.format(next_state))
                # next_state = tf.expand_dims(next_state, 0)
                # 因为后面是通过命名元组聚合,会再增加一个维度(N,),这里将state,next_state首个维度去除;
                state = state[0]
                next_state = next_state[0]
                experience = self.experience(
                    state[0], action, reward, next_state[0], done
                )
                self.memory.append(
                    experience
                )  # 注:最后一条样本因为没有next(),在env.step方法输出的state是原state
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
                    time_assumed = (time.time() - start_time) / 60
                    total_time_assumed = (time.time() - total_time) / 60
                    text = "episode:{:4d}/{},耗时:{:3.2f}分/{:3.2f},训练集: | average_treward: {:6.1f} | max_treward: {:4d} | profit_rate: {:5.3f} "
                    # \r 默认表示将输出的内容返回到第一个指针，这样的话，后面的内容会覆盖前面的内容
                    print(
                        text.format(
                            episode + 1,
                            episodes,
                            time_assumed,
                            total_time_assumed,
                            average,
                            self.max_treward,
                            profit_rate,
                        )
                    )
                    start_time = time.time()

                    break
            if self.validation:
                self.validate(episode, episodes)

            # 开始模型更新,即从样本数据中学习一次:
            if (
                    len(self.memory) > self.replay_batch_size
            ):  # memory的样本数超过batch,即开始从历史经验中学习
                self.replay(callbacks)

    def validate(self, episode, episodes):

        state = self.valid_env.reset()
        # state = tf.expand_dims(state, 0)

        for i in range(10000):
            # learn_env.act(state)有根据epsilon做随机,验证集还是直接用model输出Q值,再argmax选动作
            action = tf.math.argmax(self.Q(state), axis=-1).numpy()[0, 0]
            next_state, reward, done, info = self.valid_env.step(action)
            state = next_state
            if done:
                treward = i + 1
                profit_rate = self.valid_env.performance
                self.vperformances.append(profit_rate)
                if (episode + 1) % 10 == 0:  # 每10回合,即10个batch_size
                    text = 71 * "-"
                    text += "\nepisode:{:4d}/{},验证集: | treward: {:4d} | profit_rate: {:5.3f} |"
                    print(text.format(episode + 1, episodes, treward, profit_rate))

                break


class FinRobotAgentDQN():
    """
    DQN agent
    """

    def __init__(self, Q, Q_target, gamma=0.98, tau=1e-3, learning_rate=5e-4, learn_env=None, memory_size=4000,
                 replay_batch_size=2000, fit_batch_size=64,
                 checkpoint_path='./saved_model', checkpoint_name_prex='BTC_DQN_'):
        """
        args:
            build_model: 自建的深度学习模型,或者自定义单步训练后,做变量输入;该模型将在learn_env中训练,同一模型(训练后参数),再到valid_env中验证;
            learn_env: 装载有训练集数据,模拟训练集数据交易环境;
            replay_batch_size: 从experiences中随机replay的数量,数量为每次从经验学习的dataset数量;
            fit_batch_size: replay_batch_size中创建dataset,每次fit时,训练的batch_size;当经过replay_batch_size/fit_batch_size个epochs,全部的dataset完成一轮训练;
        """
        self.learn_env = learn_env
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = gamma
        self.tau = tau

        self.replay_batch_size = replay_batch_size
        self.fit_batch_size = fit_batch_size
        self.max_treward = 0
        self.trewards = deque(maxlen=25)
        self.averages = []
        self.performances = deque(maxlen=25)
        self.aperformances = []

        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done'])

        self.Q = Q  # Q Network model
        self.Q_target = Q_target  # Q_target Network model ;同一模型,将有不同的weights;
        self.step_num = 0  # 用于每步训练计数,计数器初始化
        self.loss = []
        self.optimizer = Adam(learning_rate=learning_rate)

        # 初始化checkpoint以存储customized train step:
        self.wait = 0  # 定义total reward mean > 200 的等待次数,满足则提前退出
        today_date = time.strftime('%y%m%d')
        checkpoint_path = checkpoint_path
        # self.ckpt = tf.train.Checkpoint(model=self.Q, optimizer=self.optimizer)
        ckpt = tf.train.Checkpoint(model=self.Q)
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            checkpoint_path,
            max_to_keep=2,
            checkpoint_name=checkpoint_name_prex + '{}'.format(today_date),
        )

    def act(self, state):
        """
        对单个state样本,执行behavior_strategy,返回action.
        """
        if tf.random.uniform((), maxval=1) <= self.epsilon:
            return self.learn_env.action_space.sample()
        # state = tf.expand_dims(state, axis=0)
        actions = self.Q(state)  # (1,lags,8)->(1,1,action_space_n) ;
        return tf.math.argmax(actions, axis=-1).numpy()[0, 0]

    def soft_update(self, ):
        """
        target_weights = tau * source_weights + (1 - tau) * target_weights
        """

        for source_weight, target_weight in zip(self.Q.trainable_variables, self.Q_target.trainable_variables):
            target_weight.assign(self.tau * source_weight + (1.0 - self.tau) * target_weight)

    @tf.function  # 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行。
    def train_step(self, experience_dataset):
        # 求导,根据导数优化变量
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(experience_dataset)
        gradients = tape.gradient(loss_value, self.Q.trainable_variables)
        # 华泰paper: 梯度范围 clamp [-1,1]
        clipped_gradients = [tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.Q.trainable_variables))

        return loss_value

    def loss_func(self, experience_dataset):
        """
        自定义loss函数,
        args:
            state_batch: 一组历史经验中,包含replay_batch_size个state的tensor,shape:(N,lags,obs_space_n)
            action_batch: shape:(N,)
            reward_batch: shape:(N,)
            next_state_batch: shape:(N,tiled_layer,obs_space_n)
            undone_batch: shape: (N,) bool类型,需要转成float才能参与算术运算;
        """
        state_batch = experience_dataset[0]
        action_batch = experience_dataset[1]
        reward_batch = experience_dataset[2]
        next_state_batch = experience_dataset[3]
        undone_batch = experience_dataset[4]

        Q_next_state = tf.math.reduce_max(
            self.Q_target(next_state_batch), axis=-1)  # (N,lags, obs_space_n)-(N,1,action_n)-> (N, 1)
        Q_next_state = tf.squeeze(Q_next_state, axis=-1)  # (N,1) - > (N,)

        TD_Q_target = reward_batch + self.gamma * Q_next_state * undone_batch
        # print('TD_Q_target.shape:{}'.format(TD_Q_target.shape))
        # Qvalue_action_pair = self.Q_target(state_batch)  # (N,action_space_n)
        Qvalue_action_pair = self.Q(state_batch)  # (N,1, action_space_n)

        # tf.gather(batch_dims=1)相当于对(N,lags,action_space_n)的首维N进行遍历循环,每个循环tf.gather:(action_space_n,)->(0,);再stack合并成(N,)
        Q_predict = tf.gather(Qvalue_action_pair,
                              indices=action_batch, axis=-1, batch_dims=1)  # (N,1)
        Q_predict = tf.squeeze(Q_predict, axis=-1)  # (N,1)-> (N,)
        # print('Q_predict.shape={}'.format(Q_predict.shape))

        loss_value = tf.keras.losses.huber(
            TD_Q_target, Q_predict)  # 据说huber对outlier loss更有效,相比mse

        return loss_value

    def replay_train_step(self, ):
        """
        使用tensorflow自定义训练的方法,自定义单步训练,loss函数中的y_true,y_predict采用模型输出指定action的Qvalue.
        从memory(历史经验),提取一次batch,学习,更新模型,相当于一个batch的训练
        args:

        """
        batch = random.sample(
            self.memory, self.replay_batch_size)  # 从memory里面,随机取出replay_batch_size个样本;

        # batch中每个样本,生成,state,action,reward,next_state,undone;再组合成dataset,包括(X_dataset,Y_dataset)

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
        # print('state_batch.shape={}'.format(state_batch.shape))

        state_batch = tf.data.Dataset.from_tensor_slices(state_batch)
        action_batch = tf.data.Dataset.from_tensor_slices(action_batch)
        reward_batch = tf.data.Dataset.from_tensor_slices(reward_batch)
        next_state_batch = tf.data.Dataset.from_tensor_slices(next_state_batch)
        undone_batch = tf.data.Dataset.from_tensor_slices(undone_batch)

        experience_dataset = tf.data.Dataset.zip(
            (state_batch, action_batch, reward_batch, next_state_batch, undone_batch)).batch(
            self.fit_batch_size).prefetch(1)

        # 训练一次,优化一次weights:
        train_loss_avg = tf.keras.metrics.Mean()  # metrics类初始化
        for experience_dataset_batch in experience_dataset:
            # 并不是模型的输出计算loss,而是loss指定action的Qvalue_action_pair
            loss_value = self.train_step(experience_dataset_batch)
            # Track progress
            train_loss_avg.update_state(loss_value)  # Add current batch loss

            # 采用soft update的方法,soft update Q target Network:
            self.soft_update()

        self.loss.append(train_loss_avg.result())
        if self.step_num % 200 == 0:
            print('step_num: {} | loss: {:.4f} | profit rate: {:6.1f} | accuracy: {:.2f}'.format(
                self.step_num, train_loss_avg.result(), self.learn_env.performance, self.learn_env.accuracy))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return train_loss_avg.result()

    def learn(self, episodes):
        """
        args:
            episodes: 训练轮数;
            callbacks: keras定义的callbacks,用列表装入多个callbacks.
        """
        # 游戏回合开始: env跑一遍;每跑一遍,生成一次样本数据,送入Memory,再学习一次;共学习episodes次
        start_time = time.time()
        total_time = time.time()
        wait = 0
        patience = 3
        best = - np.infty  # 先设置一个无穷小的数字;
        UPDATE_EVERY = 4
        for episode in range(episodes):
            state, _ = self.learn_env.reset()  # reset输出为(features,)
            # print('state:{}'.format(state))

            done = False
            treward = 0
            while not done:

                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)

                experience = self.experience(
                    tf.squeeze(state, axis=0), action, reward, tf.squeeze(next_state, axis=0),
                    done)  # 将state: (1,lags,obs_n)-> (lags,obs_n) 因为后面batch会增加第一个维度至memeory的长度;
                self.memory.append(experience)
                state = next_state
                treward += reward

                # 开始模型更新,即从batch样本数据中学习一次:
                if self.step_num % UPDATE_EVERY == 0:  # 每添加4条数据,学习一次;
                    if len(self.memory) > self.replay_batch_size:  # memory的样本数超过batch,即开始从历史经验中学习
                        # self.replay(callbacks)  # fit 方法;loss来自完整模型输出
                        train_loss = self.replay_train_step()  # 自定义loss函数,loss函数来自模型输出指定action的tensor,维度减少

                self.step_num += 1

            self.trewards.append(treward)
            average = np.mean(self.trewards)
            profit_rate = self.learn_env.performance
            self.averages.append(average)
            self.performances.append(profit_rate)
            self.aperformances.append(np.mean(self.performances))

            self.max_treward = max(self.max_treward, treward)
            time_assumed = (time.time() - start_time) / 60
            total_time_assumed = (time.time() - total_time) / 60
            text = 'episode:{:4d}/{},耗时:{:3.2f}分/{:3.2f}分,训练集: | step: {} | average_treward: {:6.1f} | max_treward: {:.4f} | average profit rate: {:6.3f}'
            if episode % 2 == 0:
                print(text.format(episode + 1, episodes, time_assumed, total_time_assumed, self.step_num, average,
                                  self.max_treward, self.aperformances[-1]))

            # 判断max_treward == t_reward,当t_reward为最大值时,将模型存盘:
            if self.trewards[-1] == self.max_treward:
                self.ckpt_manager.save()
                print('model saved @ step:{},performance:{:.3f},accuracy:{:.2f}'.format(self.step_num,
                                                                                        self.learn_env.performance,
                                                                                        self.learn_env.accuracy))

            # # 观察total reward mean 大于200的次数大约3时,提前终止训练;并每有最佳的loss值时,存盘权重
            # if mean_25 > best:
            #     best = mean_25
            #     ckpt_save_path = self.ckpt_manager.save()  # 存weight
            #     print("Saving checkpoint for episode:{} at {}".format(
            #         episode, ckpt_save_path))
            # if mean_25 > 200:
            #     wait += 1
            # if wait >= patience:
            #     ckpt_save_path = self.ckpt_manager.save()  # 存weight
            #     print(
            #         "\nepisode:{:4d}/{}, 共耗时:{:.2f}分,历{}次实现total reward 25次平均值大于200;目标实现,训练结束,weight存盘在:{}".format(
            #             episode, episodes, total_time_assumed, patience, ckpt_save_path))
            #     break

            start_time = time.time()

    def validate(self, episodes):
        for episode in range(episodes):
            state, _ = self.learn_env.reset()

            performances = []
            treward = 0
            done = False

            while not done:
                action = tf.math.argmax(self.Q(state), axis=-1).numpy()[0]
                next_state, reward, done, info = self.learn_env.step(
                    action)
                state = next_state
                treward += reward

            performances.append(treward)
            average = np.mean(performances[-5:]) / 5
            if (episode + 1) % 5 == 0:  # 每10回合,即10个batch_size
                text = 71 * '-'
                text += '\nepisode:{:4d}/{},验证集: | treward: {:.4f} | average: {:5.3f} |'
                print(text.format(episode + 1, episodes, treward, average))


class FinRobotAgentDDQN():
    """
    DDQN:
    loop for each step of episode:
        choose A from S using the policy epsilon in Q1+Q2
        Take action A, observe R,S'
        with 0.5 probability:
            Q1(S,A) <- Q1(S,A) + alpha*[R+gamma*Q2(S',argmaxQ1(S',a)) - Q1(S,A)]
        else
            Q2(S,A) <- Q2(S,A) + alpha*[R+gamma*Q1(S',argmaxQ2(S',a)) - Q2(S,A)]
        S <- S'
        until S is terminal
    """

    def __init__(self, Q, Q_target, gamma=0.98, tau=1e-3, learning_rate=5e-4, learn_env=None, memory_size=4000,
                 replay_batch_size=2000, fit_batch_size=64,
                 checkpoint_path='./saved_model', checkpoint_name_prex='BTC_DDQN_'):
        """
        args:
            build_model: 自建的深度学习模型,或者自定义单步训练后,做变量输入;该模型将在learn_env中训练,同一模型(训练后参数),再到valid_env中验证;
            learn_env: 装载有训练集数据,模拟训练集数据交易环境;
            replay_batch_size: 从experiences中随机replay的数量,数量为每次从经验学习的dataset数量;
            fit_batch_size: replay_batch_size中创建dataset,每次fit时,训练的batch_size;当经过replay_batch_size/fit_batch_size个epochs,全部的dataset完成一轮训练;
        """
        self.learn_env = learn_env
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = gamma
        self.tau = tau

        self.replay_batch_size = replay_batch_size
        self.fit_batch_size = fit_batch_size
        self.max_treward = 0
        self.trewards = deque(maxlen=25)
        self.averages = []
        self.performances = deque(maxlen=25)
        self.aperformances = []

        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done'])

        self.Q = Q  # Q Network model
        self.Q_target = Q_target  # Q_target Network model ;同一模型,将有不同的weights;
        self.step_num = 0  # 用于每步训练计数,计数器初始化
        self.loss = []
        self.optimizer = Adam(learning_rate=learning_rate)
        self.optimizer_target = Adam(learning_rate=learning_rate)

        # 初始化checkpoint以存储customized train step:
        self.wait = 0  # 定义total reward mean > 200 的等待次数,满足则提前退出
        today_date = time.strftime('%y%m%d')
        checkpoint_path = checkpoint_path
        # ckpt = tf.train.Checkpoint(model=self.Q, optimizer=self.optimizer)
        ckpt = tf.train.Checkpoint(model=self.Q)
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            checkpoint_path,
            max_to_keep=2,
            checkpoint_name=checkpoint_name_prex + '{}'.format(today_date),
        )

    def act(self, state):
        """
        对单个state样本,执行behavior_strategy,返回action.
        """
        if tf.random.uniform((), maxval=1) <= self.epsilon:
            return self.learn_env.action_space.sample()
        # state = tf.expand_dims(state, axis=0)
        actions0 = self.Q(state)  # (1,lags,8)->(1,1,action_space_n) ;
        actions1 = self.Q_target(state)  # (1,lags,8)->(1,1,action_space_n) ;
        actions = actions0 + actions1
        return tf.math.argmax(actions, axis=-1).numpy()[0, 0]

    def soft_update(self, Q1, Q2):
        """
        Q1: source
        Q2: target
        target_weights = tau * source_weights + (1 - tau) * target_weights
        """

        for source_weight, target_weight in zip(Q1.trainable_variables, Q2.trainable_variables):
            target_weight.assign(self.tau * source_weight + (1.0 - self.tau) * target_weight)

    @tf.function  # 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行。
    # 因为tf.function要求variables不能更换,所以需要分别Q1->Q2,Q2->Q1两个函数
    def train_step_Q1(self, experience_dataset):
        """
        Double_DQN,分别对Q1,以及Q2进行训练.Q1,Q2对应于不同的loss; Q1在前,实现对Q1的训练;
        :param Q1,Q2: Q,或者Q_target;
        :param experience_dataset:
        :return: loss_value
        """
        # 求导,根据导数优化变量
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(self.Q, self.Q_target, experience_dataset)
        gradients = tape.gradient(loss_value, self.Q.trainable_variables)
        # 华泰paper: 梯度范围 clamp [-1,1]
        clipped_gradients = [tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.Q.trainable_variables))

        return loss_value

    @tf.function  # 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行。
    # 因为tf.function要求variables不能更换,所以需要分别Q1->Q2,Q2->Q1两个函数
    def train_step_Q2(self, experience_dataset):
        """
        Double_DQN,分别对Q1,以及Q2进行训练.Q1,Q2对应于不同的loss; Q1在前,实现对Q1的训练;
        :param Q1,Q2: Q,或者Q_target;
        :param experience_dataset:
        :return: loss_value
        """
        # 求导,根据导数优化变量
        with tf.GradientTape() as tape:
            loss_value = self.loss_func(self.Q_target, self.Q, experience_dataset)
        gradients = tape.gradient(loss_value, self.Q_target.trainable_variables)
        # 华泰paper: 梯度范围 clamp [-1,1]
        clipped_gradients = [tf.clip_by_value(grad, clip_value_min=-1, clip_value_max=1) for grad in gradients]
        self.optimizer_target.apply_gradients(zip(clipped_gradients, self.Q_target.trainable_variables))

        return loss_value

    def loss_func(self, Q1, Q2, experience_dataset):
        """
        自定义loss函数,
        args:
            Q1,Q2: Q或者Q_target;
            state_batch: 一组历史经验中,包含replay_batch_size个state的tensor,shape:(N,lags,obs_space_n)
            action_batch: shape:(N,)
            reward_batch: shape:(N,)
            next_state_batch: shape:(N,tiled_layer,obs_space_n)
            undone_batch: shape: (N,) bool类型,需要转成float才能参与算术运算;
        """
        state_batch = experience_dataset[0]
        action_batch = experience_dataset[1]
        reward_batch = experience_dataset[2]
        next_state_batch = experience_dataset[3]
        undone_batch = experience_dataset[4]

        # argmax Q1(S',A)
        Q1_next = Q1(next_state_batch)  # (N,lags, obs_space_n)->(N,1,action_n)
        action1 = tf.math.argmax(Q1_next, axis=-1)  # (N,1,action_n)->(N,1)
        action1 = tf.squeeze(action1, axis=-1)  # (N,1)->(N,)
        # Q2(S',argmax Q1(S',A))
        Q2_next = Q2(next_state_batch)  # (N,lags, obs_space_n)->(N,1,action_n)
        Q2_predict = tf.gather(Q2_next, indices=action1, axis=-1, batch_dims=1)  # (N,1)
        Q2_predict = tf.squeeze(Q2_predict, axis=-1)  # (N,1) -> (N,)
        # R + gamma*Q2(S',argmax Q1(S',A))
        TD_Q2_target = reward_batch + self.gamma * Q2_predict * undone_batch
        # Q1(S,A)
        Q1_state = Q1(state_batch)  # (N,lags, obs_space_n)->(N,1,action_n)
        Q1_predict = tf.gather(Q1_state, indices=action_batch, axis=-1, batch_dims=1)  # (N,1)
        Q1_predict = tf.squeeze(Q1_predict, axis=-1)  # (N,1) -> (N,)

        loss_value = tf.keras.losses.huber(TD_Q2_target, Q1_predict)

        return loss_value

    def replay_train_step(self, ):
        """
        使用tensorflow自定义训练的方法,自定义单步训练,loss函数中的y_true,y_predict采用模型输出指定action的Qvalue.
        从memory(历史经验),提取一次batch,学习,更新模型,相当于一个batch的训练
        args:

        """
        batch = random.sample(
            self.memory, self.replay_batch_size)  # 从memory里面,随机取出replay_batch_size个样本;

        # batch中每个样本,生成,state,action,reward,next_state,undone;再组合成dataset,包括(X_dataset,Y_dataset)

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
        # print('state_batch.shape={}'.format(state_batch.shape))

        state_batch = tf.data.Dataset.from_tensor_slices(state_batch)
        action_batch = tf.data.Dataset.from_tensor_slices(action_batch)
        reward_batch = tf.data.Dataset.from_tensor_slices(reward_batch)
        next_state_batch = tf.data.Dataset.from_tensor_slices(next_state_batch)
        undone_batch = tf.data.Dataset.from_tensor_slices(undone_batch)

        experience_dataset = tf.data.Dataset.zip(
            (state_batch, action_batch, reward_batch, next_state_batch, undone_batch)).batch(
            self.fit_batch_size).prefetch(1)

        # 训练一次,优化一次weights:
        train_loss_avg = tf.keras.metrics.Mean()  # metrics类初始化
        for experience_dataset_batch in experience_dataset:
            if random.randint(0, 1):
                # 并不是模型的输出计算loss,而是loss指定action的Qvalue_action_pair
                loss_value = self.train_step_Q1(experience_dataset_batch)
                # Track progress
                train_loss_avg.update_state(loss_value)  # Add current batch loss
                self.soft_update(self.Q, self.Q_target)
            else:
                loss_value = self.train_step_Q2(experience_dataset_batch)
                # Track progress
                train_loss_avg.update_state(loss_value)  # Add current batch loss
                self.soft_update(self.Q_target, self.Q)

        self.loss.append(train_loss_avg.result())
        if self.step_num % 200 == 0:
            print('step_num: {} | loss: {:.4f} | profit rate: {:6.1f} | accuracy: {:.2f}'.format(
                self.step_num, train_loss_avg.result(), self.learn_env.performance, self.learn_env.accuracy))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return train_loss_avg.result()

    def learn(self, episodes):
        """
        args:
            episodes: 训练轮数;
            callbacks: keras定义的callbacks,用列表装入多个callbacks.
        """
        # 游戏回合开始: env跑一遍;每跑一遍,生成一次样本数据,送入Memory,再学习一次;共学习episodes次
        start_time = time.time()
        total_time = time.time()
        wait = 0
        patience = 3
        best = - np.infty  # 先设置一个无穷小的数字;
        UPDATE_EVERY = 4
        for episode in range(episodes):
            state, _ = self.learn_env.reset()  # reset输出为(features,)
            # print('state:{}'.format(state))

            done = False
            treward = 0
            while not done:

                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)

                experience = self.experience(
                    tf.squeeze(state, axis=0), action, reward, tf.squeeze(next_state, axis=0),
                    done)  # 将state: (1,lags,obs_n)-> (lags,obs_n) 因为后面batch会增加第一个维度至memeory的长度;
                self.memory.append(experience)
                state = next_state
                treward += reward

                # 开始模型更新,即从batch样本数据中学习一次:
                if self.step_num % UPDATE_EVERY == 0:  # 每添加4条数据,学习一次;
                    if len(self.memory) > self.replay_batch_size:  # memory的样本数超过batch,即开始从历史经验中学习
                        # self.replay(callbacks)  # fit 方法;loss来自完整模型输出
                        train_loss = self.replay_train_step()  # 自定义loss函数,loss函数来自模型输出指定action的tensor,维度减少

                self.step_num += 1

            self.trewards.append(treward)
            average = np.mean(self.trewards)
            profit_rate = self.learn_env.performance
            self.averages.append(average)
            self.performances.append(profit_rate)
            self.aperformances.append(np.mean(self.performances))

            self.max_treward = max(self.max_treward, treward)
            time_assumed = (time.time() - start_time) / 60
            total_time_assumed = (time.time() - total_time) / 60
            text = 'episode:{:4d}/{},耗时:{:3.2f}分/{:3.2f}分,训练集: | step: {} | average_treward: {:6.1f} | max_treward: {:.4f} | average profit rate: {:6.3f}'
            if episode % 2 == 0:
                print(text.format(episode + 1, episodes, time_assumed, total_time_assumed, self.step_num, average,
                                  self.max_treward, self.aperformances[-1]))

            # 判断max_treward == t_reward,当t_reward为最大值时,将模型存盘:
            if self.trewards[-1] == self.max_treward:
                self.ckpt_manager.save()
                print('model saved @ step:{},performance:{:.3f},accuracy:{:.2f}'.format(self.step_num,
                                                                                        self.learn_env.performance,
                                                                                        self.learn_env.accuracy))

            # # 观察total reward mean 大于200的次数大约3时,提前终止训练;并每有最佳的loss值时,存盘权重
            # if mean_25 > best:
            #     best = mean_25
            #     ckpt_save_path = self.ckpt_manager.save()  # 存weight
            #     print("Saving checkpoint for episode:{} at {}".format(
            #         episode, ckpt_save_path))
            # if mean_25 > 200:
            #     wait += 1
            # if wait >= patience:
            #     ckpt_save_path = self.ckpt_manager.save()  # 存weight
            #     print(
            #         "\nepisode:{:4d}/{}, 共耗时:{:.2f}分,历{}次实现total reward 25次平均值大于200;目标实现,训练结束,weight存盘在:{}".format(
            #             episode, episodes, total_time_assumed, patience, ckpt_save_path))
            #     break

            start_time = time.time()
