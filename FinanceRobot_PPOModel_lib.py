#!/usr/bin/env python
# coding: utf-8

# In[30]:

from collections import deque

import multiprocessing as mp
import time
import os
import tensorflow as tf
import tensorflow_probability as tfp  # 对模型的分布,应用sample, entropy方法
import numpy as np

from FinanceRobot_DDQNModel_lib import Decompose_FF_Linear
from FinanceRobot_Backtest_lib import Finance_Environment_V2

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # for chinese text on plt
plt.rcParams['axes.unicode_minus'] = False
# Conda多环境时,避免错误: “OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.”
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# env = Finance_Environment_V2(dataset_train, action_n=3, min_performance=0.,min_accuracy=0.1)  # 允许做空,允许大亏,使得更多的训练数据出现

class ActorModel(tf.keras.Model):
    """
    actor network.
    """

    def __init__(self, seq_len, in_features, out_features, kernel_size=25, dropout=0.3, name="Actor", **kwargs):
        """
        Initialize.
        seq_len: 输入序列长度,即lags;
        in_features: 输入预测特征数;
        pred_len: 输出序列长度
        out_features: 输出序列的特征数;
        kernel_size: moving_avg(即:AveragePooling1D)时的pool_size(窗口大小)
        """
        super(ActorModel, self).__init__(name=name, **kwargs)

        self.Decompose_FF_linear = Decompose_FF_Linear(seq_len=seq_len, in_features=in_features,
                                                       out_features=out_features, kernel_size=kernel_size,
                                                       dropout=dropout)

    def call(self, inputs: tf.Tensor):  # state: (N,lags,features)
        x = self.Decompose_FF_linear(inputs)  # (N,1,out_features)
        pi = tfp.distributions.Categorical(logits=x)  # 输出为分布

        return pi  # distribution对象


class CriticModel(tf.keras.Model):
    """
    critic network.
    """

    def __init__(self, seq_len, in_features, out_features, kernel_size=25, dropout=0.3, name="Critic", **kwargs):
        """
        Initialize.
        """
        super(CriticModel, self).__init__(name=name, **kwargs)

        self.Decompose_FF_linear = Decompose_FF_Linear(seq_len=seq_len, in_features=in_features,
                                                       out_features=out_features, kernel_size=kernel_size,
                                                       dropout=dropout)
        self.critic = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal())

    def call(self, inputs: tf.Tensor):  # state: (N,lags,features)
        x = self.Decompose_FF_linear(inputs)  # (N,1,out_features)
        critic = self.critic(x)  # (N,1,1)

        return critic  # (N,1,1)


class Step_LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    PPO baseline : frac = 1.0 - (update - 1.0) / nupdates ; update [1,updates+1]
    """

    def __init__(self, initial_lr, step_size, **kwargs):
        super(Step_LRSchedule, self).__init__(**kwargs)
        self.initial_lr = initial_lr
        self.step_size = step_size

    def __call__(self, step):
        frac = 1.0 - step / self.step_size
        return self.initial_lr * frac


# In[32]:


def worker_process(conn2, dataset, dataset_type, action_dim, trading_commission=0.001, min_performance=0.,
                   min_accuracy=0.):
    """
    conn2 : Multiprocess的Pipe的第2个控制端口,用于控制Pipe的端口2的recv和send
    env: 一个environment;conn2端口控制的对象,实现environment的reset,step,close指令;
    """
    # env = gym.make('LunarLander-v2', render_mode='rgb_array')
    env = Finance_Environment_V2(dataset, dataset_type=dataset_type, action_n=action_dim,
                                 trading_commission=trading_commission,
                                 min_performance=min_performance, min_accuracy=min_accuracy)

    while True:
        cmd, data = conn2.recv()
        # print('cmd:{}; data={}'.format(cmd, data))
        if cmd == 'reset':
            conn2.send(env.reset())
        elif cmd == 'step':
            conn2.send(env.step(data))
        elif cmd == 'get_performance':
            conn2.send(env.performance)
        elif cmd == 'get_accuracy':
            conn2.send(env.accuracy)
        elif cmd == 'close':
            # env.close() # Finance_Environment 没有close()
            conn2.close()  # 关闭子进程
            break  # 退出,否则:raise OSError("handle is closed")
        else:
            print('cmd:{},不为env接受的指定,请检查'.format(cmd))
            raise NotImplementedError


class Worker:
    def __init__(self, dataset, dataset_type, action_dim, trading_commission=0.001, min_performance=0.,
                 min_accuracy=0.):
        """"
        env: 一个environment;conn2端口控制的对象,实现environment的reset,step,close指令;
        """
        # 将可见设备设置为仅包括CPU
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # tf.config.set_visible_devices([], 'GPU')

        self.conn1, conn2 = mp.Pipe()  # conn1通过send(command,data),控制conn2;
        self.process = mp.Process(target=worker_process,
                                  args=(conn2, dataset, dataset_type, action_dim, trading_commission, min_performance,
                                        min_accuracy))
        self.process.start()


# In[ ]:


class PPO2:
    """
    本class适用于:
    1:　discrete action
    2: 收集经验data,若干episodes;使用multiprocessing运行多个worker,
    3: 经验搜集的actor,与训练的actor为同一个model,但是有不同的策略分布;故而有important sampling
    4: actor与critic为两个独立的网络;actor输出action_n个prob;critic输出value.
    """

    def __init__(self, workers, Actor, Critic, action_dim, lags, obs_dim, actor_lr=1e-4, critic_lr=1e-4, epsilon=1e-5,
                 gae_lambda=1.0, gamma=0.99, c1=0.5, c2=0.01, gradient_clip_norm=0.5,
                 n_worker=8, n_step=5, mini_batch_size=32, epochs=3, updates=50, clip_range=0.2,
                 checkpoint_path='./saved_model', checkpoint_name_prex='BTC_PPO_',
                 name='PPO2', **kwargs):
        """
        workers: 多进程Worker类
        batch_size = workers * n_step
        batches = batch_size / mini_batch_size ;
        batch_size需要是mini_batch_size的倍数;
        clip_range: PPO2中,分别对actor.log_pi,critic.value的clip参数
        gradient_clip_norm: Adam optimizer中,遏制梯度爆炸的clip_norm;

        """

        # self.env = env
        self.workers = workers
        self.action_dim = action_dim
        self.lags = lags
        self.obs_dim = obs_dim
        self.n_worker = n_worker
        self.n_step = n_step

        self.Actor = Actor
        self.Critic = Critic
        self.Actor_optimizer = tf.optimizers.Adam(
            learning_rate=actor_lr, epsilon=epsilon, global_clipnorm=gradient_clip_norm, name='Actor_optimizer')
        self.Critic_optimizer = tf.optimizers.Adam(
            learning_rate=critic_lr, epsilon=epsilon, global_clipnorm=gradient_clip_norm, name='Critic_optimizer')

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.c1 = c1
        self.c2 = c2
        self.clip_range = clip_range

        self.step = 0
        self.losses = []
        self.avg_trewards = []
        self.trewards = deque(maxlen=50)
        self.max_treward = -np.infty
        self.KL_divergence = []
        self.performance = 1
        self.accuracy = 0

        assert n_worker * n_step % mini_batch_size == 0
        self.mini_batch_size = mini_batch_size

        self.epochs = epochs
        self.updates = updates

        self.today_date = time.strftime('%y%m%d')
        checkpoint_path = checkpoint_path
        # ckpt = tf.train.Checkpoint(model=self.Actor, optimizer=self.optimizer)
        ckpt = tf.train.Checkpoint(actormodel=Actor, criticmodel=Critic)
        self.ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            checkpoint_path,
            max_to_keep=2,
            checkpoint_name=checkpoint_name_prex + '{}'.format(self.today_date),
        )

    def nworker_nstep_gae_advantage(self, values, masks, rewards):
        """
        args:
            values: shape(n_worker,n_step+1); values的长度比其它的长度多1,即n_step+ \
                          1;因为包含了next_value,是next_value与value合二为一
            masks: (n_worker,n_step)
            rewards: (n_step,,)
        outs:
            advantages: shape(n_worker,n_step)
        """
        advantage = np.zeros(shape=(self.n_worker, self.n_step + 1), dtype=np.float32)
        for t in reversed(range(self.n_step)):
            TD_error = rewards[:, t] + self.gamma * \
                       values[:, t + 1] * masks[:, t] - values[:, t]
            # print('t:{}|TD_error:{}|TD_error.shape:{}'.format(t,TD_error,TD_error.shape))
            advantage[:, t] = TD_error + self.gamma * \
                              self.gae_lambda * masks[:, t] * advantage[:, t + 1]  # (n_worker,n_step+1)

        return advantage[:, :-1]  # (n_worker,n_step)

    def compute_loss(self, cur_log_pi, old_log_pi, advantages,
                     cur_values, old_values, entropy):
        """
        1. policy_loss:
            a. ratio(theta) = exp(cur_log_pi,old_log_pi)
            b. clipped_ratio = tf.clip_by_value(min,max)
            c. normalized_advantage = advantage.normalization
            d. policy_reward = min(normalized_advantage * ratio,
                                   normalized_advantage * clipped_ratio)
            e. policy_loss = - policy_reward.mean()
        2. critic_loss:
            a. clipped_value = old_value + \
                (cur_value - old_value).clip_by_value(min,max)
            b. old_value_target = old_value + advantage
            b. critic_loss =  max[(cur_value-old_return)**2,
                                   (clipped_value-old_return)**2]
            c. critic_loss = 0.5 * critic_loss.mean()
        3. entropy_bonus:
            a: entropy = - pi * log_pi
            b: entropy.mean()
        4. total_loss = policy_loss + c1*critic_loss - c2*entropy_bonus
        args:
           cur_log_pi: 当前模型分布在采用采样数据相同的策略action下的log_pi; shape:(N,1)
           old_log_pi: 采样数据中的模型在其策略action下的log_pi; shape:(N,1)
           advantages: 采样数据中,计算出的每n_step连续数据为一组的gea_advantage;shape:(N,1)
           entropy: 当前模型分布的相对熵;
        """
        # policy_loss
        advantages = tf.stop_gradient(advantages)  # 不计算梯度
        old_log_pi = tf.stop_gradient(old_log_pi)  # 不计算梯度
        ratio = tf.exp(cur_log_pi - old_log_pi)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
        advantages_mean = tf.reduce_mean(advantages, axis=0, keepdims=True)
        advantages_std = tf.math.reduce_std(advantages, axis=0, keepdims=True)
        normalized_advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        policy_reward = tf.reduce_min(
            (normalized_advantages * ratio, normalized_advantages * clipped_ratio), axis=0)
        policy_loss = - tf.reduce_mean(policy_reward)

        # critic_loss
        clipped_values = old_values + \
                         tf.clip_by_value((cur_values - old_values), -self.clip_range, self.clip_range)
        old_value_target = old_values + advantages
        old_value_target = tf.stop_gradient(old_value_target)  # 不计算梯度
        cur_valueloss = (cur_values - old_value_target) ** 2
        clipped_valueloss = (clipped_values - old_value_target) ** 2
        critic_loss = tf.reduce_max((cur_valueloss, clipped_valueloss), axis=0)
        critic_loss = 0.5 * tf.reduce_mean(critic_loss)

        # entropy_bonus :
        total_loss = policy_loss + self.c1 * critic_loss - self.c2 * entropy

        return total_loss

    def act(self, state):
        """
        state: shape (obs_dim,);输入为单独一个state;
        """
        # state = tf.expand_dims(state, axis=0)
        pi = self.Actor(state)  # pi为分布 state: (1,lags,features) -> (1,1,action_n)
        value = self.Critic(state)  # (1,1,1)
        action = pi.sample(1)[0, 0, 0].numpy()
        action_log_prob = pi.log_prob(action)  # (1,)
        return action, action_log_prob, tf.squeeze(value, axis=[1, 2])  # (),(1,),(1,)

    def act_batch(self, states):
        """
        states: shape (n_worker,lags,obs_dim);
        """
        pi = self.Actor(states)  # pi为分布
        value = self.Critic(states)  # (n_worker,1,1)
        action = pi.sample()  # (n_worker,1)
        log_prob = pi.log_prob(action)  # (n_worker,1)
        action = tf.squeeze(action, axis=-1)  # (n_worker,1) -> (n_worker,)
        log_prob = tf.squeeze(log_prob, axis=-1)  # (n_worker,1) ->(n_worker,)
        # (n_worker),(n_worker,),(n_worker,)
        return action, log_prob, tf.squeeze(value, axis=[1, 2])

    def nworker_nstep_sampling(self, init_obs, init_treward):
        """
        n_workers(envs)的一次n_step sampling.
        args:
            init_obs: (n_worker,lags,obs_dim) 初始状态,为上一个nstep的next_obs;或者回合结束时,为env.reset()后的obs;
            init_treward: (n_worker,); 至上一个nstep累计total reward; 回合结束时,treward=0
        out:
            samples: {
                'obs':        #(n_worker,n_step+1,lags, obs_dim) -> (N,lags, obs_dim),
                'actions':    #(n_worker,n_step) -> (N,1),
                'values':     #(n_worker,n_step+1) -> (N,1),
                'log_probs':  #(n_worker,n_step) -> (N,1),
                'advantages': #(n_worker,n_step) -> (N,1)
            }
        """

        # 获得n_worker,n_step 诸状态numpy array:

        obs = np.zeros(shape=(self.n_worker, self.n_step + 1, self.lags, self.obs_dim), dtype=np.float32)
        actions = np.zeros(shape=(self.n_worker, self.n_step), dtype=np.int32)
        rewards = np.zeros(shape=(self.n_worker, self.n_step), dtype=np.float32)
        masks = np.zeros(shape=(self.n_worker, self.n_step), dtype=np.float32)
        values = np.zeros(shape=(self.n_worker, self.n_step + 1), dtype=np.float32)
        log_probs = np.zeros(shape=(self.n_worker, self.n_step), dtype=np.float32)

        # treward = np.zeros(shape=(self.n_worker,),dtype=np.float32)
        # for w, worker in enumerate(self.workers):
        obs[:, 0] = init_obs
        treward = init_treward
        for t in range(self.n_step):
            action, log_prob, value = self.act_batch(
                obs[:, t])  # (n_worker),(n_worker,),(n_worker,)

            actions[:, t] = action
            log_probs[:, t] = log_prob
            values[:, t] = value

            for w, worker in enumerate(self.workers):
                worker.conn1.send(('step', actions[w, t]))
            for w, worker in enumerate(self.workers):
                next_obs, reward, done, info = worker.conn1.recv()
                next_obs = tf.squeeze(next_obs, axis=0)  # (1,lags,obs_dim)-> (lags,obs_dim)
                mask = 1 - done

                obs[w, t + 1] = next_obs  # (lags,obs_dim)
                rewards[w, t] = reward
                masks[w, t] = mask

                treward[w] += reward

                self.step += 1
                if done:
                    worker.conn1.send(('reset', None))
                    next_obs, _ = worker.conn1.recv()
                    next_obs = tf.squeeze(next_obs, axis=0)  # (1,lags,obs_dim)-> (lags,obs_dim)
                    obs[w, t + 1] = next_obs  # (lags,obs_dim)

                    self.trewards.append(treward[w])
                    avg_treward = np.mean(self.trewards)
                    self.avg_trewards.append(avg_treward)
                    self.max_treward = max(self.max_treward, treward[w])
                    if t == self.n_step - 1:
                        print(
                            f"step: {self.step} | worker_{w}@n_step_{t}: average total_reward after train data exhaustion : {avg_treward:.1f} | max total_reward: {self.max_treward:.1f}")

                    treward[w] = 0

        init_obs = obs[:, t + 1]
        init_treward = treward

        # 计算advantage;
        next_value = self.Critic(obs[:, t + 1])  # (N,1,1)
        values[:, t + 1] = tf.squeeze(next_value, axis=[1, 2])  # (N,1,1) -> (N,)
        advantages = self.nworker_nstep_gae_advantage(values, masks, rewards)

        # 获得values,advantages,log_probs 字典,以便于积累追加. numpy array占用内存大;
        # (n_worker,n_step+1,obs_dim)->(N,obs_dim)
        obs = tf.convert_to_tensor(
            obs[:, :-1].reshape((-1, self.lags, self.obs_dim)), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions.reshape(
            (-1, 1)), dtype=tf.int32)  # (n_worker,n_step) -> (N,1)
        # (n_worker,n_step+1) -> (N,1)
        values = tf.convert_to_tensor(values[:, :-1].reshape((-1, 1)), dtype=tf.float32)
        log_probs = tf.convert_to_tensor(log_probs.reshape(
            (-1, 1)), dtype=tf.float32)  # (n_worker,n_step) -> (N,1)
        advantages = tf.convert_to_tensor(advantages.reshape(
            (-1, 1)), dtype=tf.float32)  # (n_worker,n_step) -> (N,1)

        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_probs': log_probs,
            'advantages': advantages
        }

        return samples, init_obs, init_treward

    def samples_dataset(self, samples, ):
        """
        args:
            samples: {
                'obs':        #(n_worker,n_step+1, lags, obs_dim) -> (N, lags, obs_dim),
                'actions':    #(n_worker,n_step) -> (N,1),
                'values':     #(n_worker,n_step+1) -> (N,1),
                'log_probs':  #(n_worker,n_step) -> (N,1),
                'advantages': #(n_worker,n_step) -> (N,1)
            }
        out: 
            dataset: 

        """

        datasets = []
        for k, v in samples.items():
            dataset = tf.data.Dataset.from_tensor_slices(v)
            datasets.append(dataset)
        exp_dataset = tf.data.Dataset.zip(tuple(datasets)).shuffle(
            self.n_worker * self.n_step + 100).batch(self.mini_batch_size).prefetch(1)

        return exp_dataset

    @tf.function
    def step_train(self, old_obs_batch, old_action_batch, old_value_batch, old_log_prob_batch, old_advantage_batch):
        with tf.GradientTape() as tape:
            pi = self.Actor(old_obs_batch)
            cur_value_batch = self.Critic(old_obs_batch)  # (N,1)
            cur_log_prob = pi.log_prob(tf.squeeze(
                old_action_batch, axis=-1))  # (N,)
            cur_log_prob = tf.expand_dims(cur_log_prob, axis=-1)  # (N,)-> (N,1)
            entropy = tf.reduce_mean(pi.entropy())
            loss = self.compute_loss(
                cur_log_prob, old_log_prob_batch, old_advantage_batch, cur_value_batch, old_value_batch, entropy)
            # print('loss:{}'.format(loss))

        Actor_Critic_weights = self.Actor.trainable_variables + self.Critic.trainable_variables
        gradients = tape.gradient(loss, Actor_Critic_weights)
        n_actor_vars = len(self.Actor.trainable_variables)
        Actor_gradients = gradients[:n_actor_vars]
        Critic_gradients = gradients[n_actor_vars:]
        self.Actor_optimizer.apply_gradients(
            zip(Actor_gradients, self.Actor.trainable_variables))
        self.Critic_optimizer.apply_gradients(
            zip(Critic_gradients, self.Critic.trainable_variables))

        approx_kl_divergence = .5 * ((old_log_prob_batch - cur_log_prob) ** 2)
        approx_kl_divergence = tf.reduce_mean(approx_kl_divergence)

        return loss, approx_kl_divergence

    def nworker_nstep_train(self, samples, ):
        """
        1. samples - > experience_dataset;
        2. 获得 old_log_pi=exp_dataset[3], advantages=exp_dataset[4], old_values=exp_dataset[2],old_action=exp_dataset[1]
        3. 运行模型,获得当前的: cur_log_pi,cur_values
        4. 求梯度,及反向传播
        args:
            samples: {
                'obs':        #(n_worker,n_step+1,obs_dim) -> (N,obs_dim),
                'actions':    #(n_worker,n_step) -> (N,1),
                'values':     #(n_worker,n_step+1) -> (N,1),
                'log_probs':  #(n_worker,n_step) -> (N,1),
                'advantages': #(n_worker,n_step) -> (N,1)
            }
        """
        loss_metric = tf.keras.metrics.Mean()
        KL_metric = tf.keras.metrics.Mean()
        for epoch in range(self.epochs):
            exp_dataset = self.samples_dataset(samples)
            # exp_dataset = self.samples_dataset_origin(samples)
            for exp in exp_dataset:
                old_obs_batch = exp[0]  # (mini_batch,obs_dim)
                old_action_batch = exp[1]  # (mini_batch,1)
                old_value_batch = exp[2]  # (mini_batch,1)
                old_log_prob_batch = exp[3]  # (mini_batch,1)
                old_advantage_batch = exp[4]  # (mini_batch,1)

                loss, approx_kl_divergence = self.step_train(
                    old_obs_batch, old_action_batch, old_value_batch, old_log_prob_batch, old_advantage_batch)

                loss_metric.update_state(loss)
                KL_metric.update_state(approx_kl_divergence)

            self.losses.append(loss_metric.result())
            self.KL_divergence.append(KL_metric.result())

            # self.Actor_optimizer.zero_grad()
            # self.Critic_optimizer.zero_grad()
            loss_metric.reset_states()
            KL_metric.reset_states()

        # avg_treward = self.avg_trewards[-1] if self.avg_trewards else -np.infty
        # loss = self.losses[-1] if self.losses else -np.infty
        # if self.step % 100 == 0:
        #     text = 'step: {:5d} | avg_treward: {:5.1f} | max_treward: {:.1f} | loss: {:9.2f}'
        #     print(text.format(self.step, avg_treward,
        #           self.max_treward, loss))

    def nworker_nstep_training_loop(self, updates=50):
        # today_date = time.strftime('%y%m%d')
        start_time = time.time()
        wait = 0
        patience = 3
        best = - np.infty  # 先设置一个无穷小的数字;

        init_obs = np.zeros(shape=(self.n_worker, self.lags, self.obs_dim), dtype=np.float32)
        init_treward = np.zeros(shape=(self.n_worker,))
        for w, worker in enumerate(self.workers):
            worker.conn1.send(('reset', None))
        for w, worker in enumerate(self.workers):
            init_state, _ = worker.conn1.recv()
            init_state = tf.squeeze(init_state, axis=0)  # (1,lags, obs_dim) -> (lags, obs_dim)
            init_obs[w] = init_state

        for update in range(updates):
            update_start_time = time.time()

            samples, init_obs, init_treward = self.nworker_nstep_sampling(init_obs, init_treward)
            self.nworker_nstep_train(samples)

            avg_treward = self.avg_trewards[-1] if self.avg_trewards else -np.infty
            if (update + 1) % 25 == 0:
                time_spend = (time.time() - update_start_time) / 60
                t_time_spend = (time.time() - start_time) / 60
                loss = self.losses[-1] if self.losses else -np.infty

                # 输出和打印env.performance和env.accuracy:
                for w, worker in enumerate(self.workers):
                    worker.conn1.send(('get_performance', None))
                for w, worker in enumerate(self.workers):
                    self.performance = worker.conn1.recv()
                for w, worker in enumerate(self.workers):
                    worker.conn1.send(('get_accuracy', None))
                for w, worker in enumerate(self.workers):
                    self.accuracy = worker.conn1.recv()
                # print(f"{self.step}: performance: {self.performance} | accuracy: {self.accuracy}")

                text = 'update:{:3d}/{}, 耗时:{:3.2f}分/{:3.2f}分 | step: {:5d} | performance: {:.1f} | accuracy: {:.2f} | loss: {:3.2f}'
                print(text.format(update + 1, updates, time_spend, t_time_spend,
                                  self.step, self.performance, self.accuracy, loss))

            # 观察total reward mean 大于200的次数大约3时,提前终止训练;并每有最佳的loss值时,存盘权重
            if avg_treward > best:
                best = avg_treward
                actor_save_path = './saved_model/BTC_PPO_actor_{}_{}.h5'.format(
                    updates, self.today_date)
                critic_save_path = './saved_model/BTC_PPO_critic_{}_{}.h5'.format(
                    updates, self.today_date)
                self.Actor.save_weights(
                    actor_save_path, overwrite=True, save_format='h5')
                self.Critic.save_weights(
                    critic_save_path, overwrite=True, save_format='h5')
                self.ckpt_manager.save()
                print("Saving PPO weights in both H5 format and checkpoint @ update:{} ".format(update + 1))

    def close_process(self, ):
        for w, worker in enumerate(self.workers):
            worker.conn1.send(('close', None))

# if __name__ == "__main__":
#     action_dim = 4
#     obs_dim = 8
#     n_worker = 16
#     n_step = 5
#     mini_batch_size = 5  # int(n_worker * n_step / 4)
#     epochs = 3
#     updates = 5000
#     Actor = ActorModel(num_actions=action_dim)
#     Critic = CriticModel()
#
#     # ---Load 最近一次 存储的weights:(需要model call一次,才能够load)
#     # env = gym.make('LunarLander-v2', continuous=False, render_mode='rgb_array')
#     # init_obs, _ = env.reset()
#     # init_obs = tf.expand_dims(init_obs, axis=0)
#     # Actor(init_obs)
#     # Critic(init_obs)
#     # Actor.load_weights('./saved_model/PPO_actor_{}_{}.h5'.format(
#     #     '15000', '230516'))
#     # Critic.load_weights('./saved_model/PPO_critic_{}_{}.h5'.format(
#     #     '15000', '230516'))
#     # env.close()
#     # # --------------
#     #
#     # step_lr = Step_LRSchedule(1e-4, updates)
#     workers = []
#     for i in range(n_worker):
#         worker = Worker()
#         workers.append(worker)
#     PPO_agent = PPO2(workers, Actor, Critic, action_dim, obs_dim, actor_lr=1e-4, critic_lr=5e-04, gae_lambda=0.99,
#                      gamma=0.98,
#                      c1=1., gradient_clip_norm=10., n_worker=n_worker, n_step=n_step, epochs=epochs,
#                      mini_batch_size=mini_batch_size)
#     PPO_agent.nworker_nstep_training_loop(updates)
#
#     today = time.strftime("%y%m%d")
#     fig, ax = plt.subplots(1, 3, figsize=(20, 6), )
#     ax[0].plot(PPO_agent.avg_trewards)
#     ax[0].set_title('average total reward')
#     ax[1].plot(PPO_agent.losses)
#     ax[1].set_title('PPO_loss')
#     ax[2].plot(PPO_agent.KL_divergence)
#     ax[2].set_title('approximal KL_divergence')
#     fig.savefig('./saved_model/PPO_{}_w{}_s{}_b{}_u{}'.format(today,
#                                                               n_worker, n_step, mini_batch_size, updates))
#
#     PPO_agent.close_process()
