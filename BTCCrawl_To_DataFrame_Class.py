#!/usr/bin/env python
# coding: utf-8

# ## **准备数据: 爬取,清洗**
# - 从网站上爬取BTC交易数据;
# - 计算5个指标,生成DataFrame; 注: 以后可以增加多个指标;

# In[46]:

from binance.spot import Spot
import datetime
import json
# import time
import sys

import numpy as np
import pandas as pd
import requests
from IPython.display import display
from configparser import ConfigParser


# 注: 可以使用的exchangeId (exchange=huobi)有:
# ["alterdice" "bibox" "bigone" "bilaxy" "binance" "binanceus" "bitfinex" "bithumbglobal" "bitmax" "bitso" #"bitstamp"
# "btcturk" "coinex" "coinsbit" "cointiger" "crypto" "currency-com" "dex-trade" "digifinex" "exmarkets" #"exmo" "gate"
# "gdax" "hitbtc" "huobi" "indodax" "kickex" "kraken" "kucoin" "kuna" "lbank" "max-exchange" #"mercatox" "okcoin"
# "paribu" "poloniex" "probit" "qryptos" "quoine" "therocktrading" "tidex" "wazirx" "whitebit" #"zb"  ]
# poloniex交易所到2022年8月1日数据停止

# Folder_base = "E:/Python_WorkSpace/量化交易/股票分析/"
# URL = "https://api.coincap.io/v2/candles?exchange=huobi&interval=h12&baseId=bitcoin&quoteId=tether"
# StartDate = "2022-06-15"
# EndDate = "2022-10-01"
# BTC_json = "BTC_h12.json"


def Datetime2Timstamp13bit(_datetime):
    """
    args:
        datetime: text that pandas could accept
    out:
        timestamp : 13bit
    """
    timestamp = pd.Timestamp(_datetime, tz='UTC', unit='ns').value
    timestamp = int(timestamp / 1000000)
    return timestamp


def get_api_key(config_file_path):
    """
    从配置文件config.ini中,读取api_key,api_secret;避免程序代码中明文显示key,secret.
    args:
        config_file_path: config.ini的文件路径(包含文件名,即: directory/config.ini)
    out:
        两个str,分别是ape_key,api_secret
    """
    config = ConfigParser()
    config.read(config_file_path, encoding="utf-8")  # utf-8支持中文
    return config["keys"]["api_key"], config["keys"]["api_secret"]


class BTC_data_acquire:
    def __init__(
            self,
            URL,
            StartDate,
            EndDate,
            Directory,
            BTC_json='BTC_h12.json',
            BinanceBTC_json='BinanceBTC_h12.json',
            binance_api_key=None,
            binance_api_secret=None
    ):
        """
        从BTC网站上爬取数据,然后生成DataFrame,再计算RSI,合并成DataFrame
            Args:
                URL : 爬取网站的API地址
                StartDate : 爬取数据的起始时间
                EndDate: 爬取数据的结束时间
                Directory: BTC数据存储json文件的存储目录
                BTC_json: BTC数据存储json文件名称(coincap API)
                BinanceBTC_json:BTC数据存储json文件名称(Binance API)
                binance_api_key: optinal,当使用Binance API通道时,api_key
                binance_api_secret: optinal,当使用Binance API通道时,api_secret
            Returns:
                输出DataFrame格式的包含各个技术指标的数据;
        """
        self.URL = URL
        self.StartDate = StartDate
        self.EndDate = EndDate
        self.Directory = Directory
        self.BTC_json = BTC_json
        self.BinanceBTC_json = BinanceBTC_json
        self.binance_api_key = binance_api_key
        self.binance_api_secret = binance_api_secret

    def BinanceAPI_2_DF(self, FromWeb, interval='12h'):
        """
        通过Binance API接口,爬取交易数据,然后生成DataFrame,与已有json历史数据合并,生成自有记录开始的完整的Data
        input:
            FromWeb: bool; True表示从URL上爬取,False表示从json上读取
            interval:  (str) – the interval of kline, e.g 1m, 5m, 1h, 1d, etc.
            api_key: Binance API key
            api_secret: Binance API secret
        Returns:
            DataFrame.index: open_time, 时间戳;DataFrame.clolums:
            [open,high,low,close,volume,amount,num_trades,bid_volume,bid_amount]
        """
        # 从硬盘上读取以历史BTC数据
        # Binance API是基于UTC时区,在转换成时间戳于json写入再读出,失去了时区属性;'
        Data = pd.read_json(self.Directory + self.BinanceBTC_json)
        # 1)先恢复UTC时区属性 2) 与爬取数据合并后,再转换到Asia/Shanghai时区
        Data.index = Data.index.tz_localize('UTC')

        if FromWeb:

            client = Spot(
                api_key=self.binance_api_key,
                api_secret=self.binance_api_secret,
                base_url=self.URL)

            ping = client.ping()

            if ping == {}:
                print('Binance Rest API 联通测试通过 !')
            else:
                print(ping)
                print('Binance Rest API 联通测试失败,请检查网络,或者VPN, (注:美国的IP地址暂不为接受)')
                sys.exit(0)

            StartTime = Datetime2Timstamp13bit(self.StartDate)
            EndTime = Datetime2Timstamp13bit(self.EndDate)
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'startTime': StartTime,
                'endTime': EndTime,
                'limit': 1000,
            }

            response = client.klines(**params)

            data = pd.DataFrame(response)
            data.set_index(data[0].apply(lambda x: pd.Timestamp(
                x, unit='ms', tz='UTC')), inplace=True)  # open_ts变换成UTC时间,并设成datatimeindex;

            # 实际上,这里的UTC时区,存入json文件,再被读出时,时区会被忽略,时间戳,变成native/localized时间,失去了时区属性

            # 第0列:开盘时间;第6列:收盘时间;第11列,可忽略列;一并删除.
            data.drop(columns=[0, 6, 11], inplace=True)

            columns_dict = {1: "open", 2: "high", 3: "low", 4: "close", 5: "volume",  # 6: "close_ts"
                            7: "amount", 8: "num_trades", 9: "bid_volume", 10: "bid_amount"}

            data.rename(columns=columns_dict, inplace=True)  # 更列名;
            data.index.rename('open_time', inplace=True)  # 更index名;
            # --------------------------------------------------------

            Data = pd.concat([Data, data], join='inner')
            # 非常漂亮的删除重复Index行(并且有选择的保留first行,或者last行)的方法,
            Data = Data.loc[~Data.index.duplicated(keep='last')]

            # 将数据转换成float16,用于避免当不经过写入读取Json文件进行d_return计算,Data['close'] / Data['close'].shift()会导致运算失败
            Data = Data[Data.columns[:]].astype('float32')
            print('从web中读出Data.info:{}'.format(Data.info()))

            # 因为合并后的数据,有可能出现index的时间戳穿插;即后爬取的时间段比json存储的时间段还要早,导致爬取的早先时间段会放在json较后的时间段之后的情况,所以这里排序一下
            data.index = data.index.sort_values()  # sort方法缺省升序排列

            # 爬取的最新数据,与硬盘上历史数据合并后,重新存储Json
            Data.to_json(self.Directory + self.BinanceBTC_json)
            print('BTC数据合并存入:{}'.format(self.Directory + self.BinanceBTC_json))
        else:
            print('从json中读出Data.info:{}'.format(Data.info()))

        # print('有重复Index的行共有:{}'.format(len(Data.index[Data.index.duplicated()])))

        # ------------------------------------------------------------

        Data.index = Data.index.tz_convert(
            'Asia/Shanghai')  # 1) 上面已经基于UTC时区, Data合并, 2)再将时区从UTC更改到上海时间

        return Data

    def crawl_DataFrame(
            self,
    ):
        """
        从BTC网站上爬取数据,输出网站原始数据不予处理.
            Args:

            Returns:
                输出DataFrame格式的原始数据;不做任何处理,DataFrame.columns:[period,open,high,low,close,volume]

        """

        # 爬取BTC价格:
        # Timestamp = str(time.time_ns())[:13]

        start = str(
            datetime.datetime.timestamp(
                datetime.datetime.strptime(self.StartDate, "%Y-%m-%d")
            )
            * 1000
        )[:13]
        end = str(
            datetime.datetime.timestamp(
                datetime.datetime.strptime(self.EndDate, "%Y-%m-%d")
            )
            * 1000
        )[:13]
        URL = self.URL + "&start=" + str(start) + "&end=" + str(end)
        headers = {'Accept-Encoding': 'gzip',
                   "Authorization": "Bearer 8a229699-6244-4154-b12d-9fe8197b4fe1"}

        # 构造请求
        response = requests.get(URL, headers=headers)
        Candles = json.loads(response.text)
        Candle = pd.DataFrame(Candles["data"])
        print("爬取数据DataFrame.shape: {}".format(Candle.shape))
        # print(response)
        return Candle

    def GenDF_Frjson_FrWeb(self, FromWeb):
        """
        从BTC网站上爬取数据,然后生成DataFrame,与已有json历史数据合并,生成自有记录开始的完整的Data
        input:
            FromWeb: bool; True表示从URL上爬取,False表示从json上读取        
        Returns:
            DataFrame.index: 时间戳;DataFrame.clolums: [open,high,low,close,volumn]
        """
        # 从硬盘上读取以历史BTC数据
        Data = pd.read_json(self.Directory + self.BTC_json)

        if FromWeb:
            Candle = self.crawl_DataFrame()
            # 将'period'转化为时间戳的datetime格式,然后设定其为Index (行号)
            Candle['period'] = Candle['period'].astype('datetime64[ms]')
            Candle.set_index('period', inplace=True)  # 将Period列变成Index

            # display(Data.head(2))
            # display(Data.tail(2))
            # Data.info()
            # 更新(添加)存储的Json文件数据
            # Data_update = pd.read_json(Folder_base+'BTC_h8_20210825_20210903.json')
            # 添加 上面API读出的数据: Candle
            Data_update = Candle
            Data = pd.concat([Data, Data_update], join='inner')
            # 非常漂亮的删除重复Index行(并且有选择的保留first行,或者last行)的方法,
            Data = Data.loc[~Data.index.duplicated(keep='last')]

            # 将数据转换成float16,用于避免当不经过写入读取Json文件进行d_return计算,Data['close'] / Data['close'].shift()会导致运算失败
            Data = Data[Data.columns[:]].astype('float32')
            # display(Data.head(2))
            # display(Data.tail(2))
            print('从web中读出Data.info:{}'.format(Data.info()))

            # 爬取的最新数据,与硬盘上历史数据合并后,重新存储Json
            Data.to_json(self.Directory + self.BTC_json)
            print('BTC数据合并存入:{}'.format(self.Directory + self.BTC_json))
        else:
            print('从json中读出Data.info:{}'.format(Data.info()))

        # display(Data.head(2))
        # display(Data.tail(2))
        # print('有重复Index的行共有:{}'.format(len(Data.index[Data.index.duplicated()])))

        return Data

    def MarketFactor_DataFrame(self, FromWeb=True, Market_Factor=True, weekdays=7, DayH=2):
        """
        从存储位置读出json,然后生成DataFrame,再计算RSI,合并成DataFrame;
        除基础的指标: ['d_return', 'd_amplitude', 'volume', 'RSI14', 'high', 'low', 'open','close']之外,
        新增加指标有:
            # log{Today's Close Price - (T-5) day's close price};
            # log{high/low};
            # log{Today's Open Price - (T-5) day's open price};
            # Close/Previous Close;
            # Open/Previous Close;
            # High/Previous Close;
            # Low/Previous Close
            Args:
            FromWeb: 是否从网站上爬取(True);或者从存储的json文件中读取;
            DayH: H12代表12小时的数据,日24小时,则每两个序列表示一天;DayH=24/12
            Market_Factor: 是否输出新增加的Market_Factor指标;
            weekdays: 一周有几天;周末休市是5天;全天候是7天;
            Returns:
                输出DataFrame格式的包含各个技术指标的数据:['d_return', 'd_amplitude', 'volume', 'RSI14', 'high', 'low', 'open', 'close']
        """
        Data = self.GenDF_Frjson_FrWeb(FromWeb=FromWeb)

        # 日收益率
        Data['d_return'] = Data['close'] / Data['close'].shift() - 1  # Dt / D(t-1) - 1
        # 日振幅:
        Data['d_amplitude'] = Data['high'] - Data['low']

        # log{Today's Close Price - (T-5) day's close price};
        Data['Log_close_weeklag'] = np.log(
            Data['close'] / Data['close'].shift(periods=weekdays))
        # log{high/low};
        Data['Log_high_low'] = np.log(Data['high'] / Data['low'])
        # log{Today's Open Price - (T-5) day's open price};
        Data['Log_open_weeklag'] = np.log(
            Data['open'] / Data['open'].shift(periods=weekdays))

        # Open/Previous Close;
        Data['open_pre_close'] = Data['open'] / Data['close'].shift()
        # High/Previous Close;
        Data['high_pre_close'] = Data['high'] / Data['close'].shift()
        # Low/Previous Close
        Data['low_pre_close'] = Data['low'] / Data['close'].shift()

        # H12代表12小时的数据,24小时,则每两个序列表示一天
        # DayH = 24/12
        RSI_period = int(weekdays * DayH)
        RSI = self.RSI(Data['close'], periods=RSI_period)
        RSI_columnName = 'RSI_' + str(weekdays)
        Data[RSI_columnName] = RSI

        # 定义观测数据X
        if Market_Factor:
            X = Data[['d_return', 'd_amplitude', 'volume', RSI_columnName, 'Log_close_weeklag',
                      'Log_high_low', 'Log_open_weeklag', 'open_pre_close', 'high_pre_close',
                      'low_pre_close', 'high', 'low', 'open', 'close']]
            # X.rename(columns={0:'d_amplitude','close':'d_return'},inplace=True)

        else:
            # X = pd.concat([d_return,d_amplitude,Data['volume'],RSI],axis=1)
            X = Data[['d_return', 'd_amplitude', 'volume',
                      RSI_columnName, 'high', 'low', 'open', 'close']]

        X = X[RSI_period:]
        # display(X.describe())
        # # type(X)
        display(X.head(2))
        display(X.tail(2))

        return X

    def Temporal_MarketFactor_DataFrame(self, FromWeb=True, Market_Factor=True, weekdays=7,
                                        DayH=2, minute_level=False):
        """
        从存储位置读出json,然后生成DataFrame,再计算RSI,再将year/month/day/weekday/hour/minute数据分别对应成0-5列,最后,合并成DataFrame
            Args:
            FromWeb: bool,是否从网站上爬取(True);或者从存储的json文件中读取;
            DayH: H12代表12小时的数据,日24小时,则每两个序列表示一天;DayH=24/12
            minute_level: bool,是否将分钟作为第5列;False时,时间标记,只包含0-4列,即:year/month/day/weekday/hour;
            Returns:
                输出DataFrame格式的包含各个技术指标的数据,其中前0->4列(minute_level=False)为year/month/day/weekday/hour,或者前(0-5)列(minute_level=True),为year/month/day/weekday/hour/minute;
        """

        X = self.MarketFactor_DataFrame(FromWeb=FromWeb, Market_Factor=Market_Factor,
                                        weekdays=weekdays, DayH=DayH)
        # 添加时间标签列,分别为 year/month/day/weekday/hour/minute,对应于第0-5列
        X['year'] = X.index.year
        X['month'] = X.index.month
        X['day'] = X.index.day
        X['weekday'] = X.index.weekday
        X['hour'] = X.index.hour

        if minute_level:
            X['minute'] = X.index.minute
            # 将year/month/day/weekday/hour/minute列,转移到最前面
            X_columns = X.columns[(X.columns.shape[0] - 6):].tolist() + X.columns[:(X.columns.shape[0] - 6)].tolist()
            X = X[X_columns]
        else:  # 否则,抛弃minute,到hour为止;
            # 将year/month/day/weekday/hour列,转移到最前面
            X_columns = X.columns[(X.columns.shape[0] - 5):].tolist() + \
                        X.columns[:(X.columns.shape[0] - 5)].tolist()
            X = X[X_columns]

        display(X.describe())
        # type(X)
        display(X.head(2))
        display(X.tail(2))

        return X

    def RSI(self, t, periods=10):
        """
        # 计算RSI
        #RS=X天的平均上涨点数/X天的平均下跌点数,RSI=100×RS/(1+RS)
        #UP_AVG = UP_AMOUNT/PERIODS (周期内上涨数量平均)
        #UP_AVG = (UP_AVG_PREV * (PERIODS - 1) + UP) / PERIODS
        #DOWN_AVG = DOWN_AMOUNT/PERIODS（周期内下跌数量平均）
        #DOWN_AVG = (UP_AVG_PREV * (PERIODS - 1) + DOWN) / PERIODS
        #RS = UP_AVG/DOWN_AVG（相对平均）
        #RSI  = 100 - 100 / (1 + RS)  （相对强弱指数）

        Args:
            t : 计算RSI的数据列,为1维DataFrame系列;
            periods: RSI观察的窗口值,缺省为10;
        Returns:
            rsies: RSI数据列;
        """
        length = len(t)
        rsies = [np.nan] * length
        # 数据长度不超过周期，无法计算；
        if length <= periods:
            return rsies
        # 用于快速计算；
        up_avg = 0
        down_avg = 0

        # 首先计算第一个RSI，用前periods+1个数据，构成periods个价差序列;
        first_t = t[:periods + 1]
        for i in range(1, len(first_t)):
            # 价格上涨;
            if first_t[i] >= first_t[i - 1]:
                up_avg += first_t[i] - first_t[i - 1]
            # 价格下跌;
            else:
                down_avg += first_t[i - 1] - first_t[i]
        up_avg = up_avg / periods
        down_avg = down_avg / periods
        rs = up_avg / down_avg
        rsies[periods] = 100 - 100 / (1 + rs)

        # 后面的将使用快速计算；
        for j in range(periods + 1, length):
            # up = 0
            # down = 0
            if t[j] >= t[j - 1]:
                up = t[j] - t[j - 1]
                down = 0
            else:
                up = 0
                down = t[j - 1] - t[j]
            # 类似移动平均的计算公式;
            up_avg = (up_avg * (periods - 1) + up) / periods
            down_avg = (down_avg * (periods - 1) + down) / periods
            rs = up_avg / down_avg
            rsies[j] = 100 - 100 / (1 + rs)
        return rsies

    def SingleCloseCol_addfeatures(self, FromWeb, close_colName, lags=5, window=20):
        """
        只观察股票数据中的收盘价,并将收盘价做以下处理,生成各个特征:
        log_return: np.log(df/df.shift()) 相邻时序收盘价格比值,取对数,反应收益;
        Roll_price_sma: 收盘价Rolling指定window后,对window内取mean,即: simple moving average;
        Roll_price_min: 收盘价Rolling指定window后,对window内取min,即: 滚动最小值;
        Roll_price_max: 收盘价Rolling指定window后,对window内取max,即: 滚动最大值;
        Roll_return_momt: log_return的Rolling指定window后,对window内取mean,即: 动量;
        Roll_return_std: log_return的Rolling指定window后,对window内取std,即: 滚动波动率;
        Mark_return : log_return正值取1,负值取0;即:收益正负标记;
        以上7列,再每列滑动lags时序,生成lags*7列,列名: lag1_columnname.
        最后1列,为收盘价列;列名:close_colName;

        input:
            data: 股票数据DataFrame,包含有收盘价的列,列名为:'close_colName' (例如:'close');
            close_colName: 股票数据中收盘价的列名;
            lags: 延时滑动的时序数;
            window: Rolling的window大小

        """
        data = self.GenDF_Frjson_FrWeb(FromWeb=FromWeb)
        df = pd.DataFrame(data[close_colName])  # 原DataFrame中,除收盘价以外的其它列舍弃;
        df.dropna(inplace=True)
        df["log_return"] = np.log(
            df[close_colName] / df[close_colName].shift())  # 收益率(对数)
        df["Roll_price_sma"] = (
            df[close_colName].rolling(window).mean()
        )  # simple moving average
        df["Roll_price_min"] = (
            df[close_colName].rolling(window).min()
        )  # 滚动最小值与滚动最大值、动量以及滚动波动率
        df["Roll_price_max"] = df[close_colName].rolling(window).max()  # 滚动最大值;
        df["Roll_return_mom"] = df["log_return"].rolling(window).mean()  # 动量;
        df["Roll_return_std"] = df["log_return"].rolling(window).std()  # 滚动波动率
        df.dropna(inplace=True)
        df["Mark_return"] = np.where(df["log_return"] > 0, 1, 0)
        features = [
            "log_return",
            "Roll_price_sma",
            "Roll_price_min",
            "Roll_price_max",
            "Roll_return_mom",
            "Roll_return_std",
            "Mark_return",
            close_colName,
        ]
        df = df[features]
        for f in features:
            for lag in range(1, lags + 1):  # 包括收盘价,Mark_return每列都延时lags次;
                col = "{}_lag{}".format(f, lag)
                df[col] = df[f].shift(lag)
                df.dropna(inplace=True)

        display(df.describe())
        # type(X)
        display(df.head(2))
        display(df.tail(2))
        return df

    def MarketFactor_ClosePriceFeatures(self, by_BinanceAPI, FromWeb, close_colName='close', lags=5, window=20,
                                        horizon=5, interval='12h', DayH=2, MarketFactor=True, weekdays=7):
        """
        实现: 1) 收集收盘价以及衍生特征,再加上OHLC关键特征; 2) 鉴于有文章发现reward函数,在预测后5个交易日比预测第二个交易日效果更好,
        增加horizon参数,增添一个log_return特征(避免标准化/归一化),来表示horizon之后的t_horizon时刻的收盘价与t时刻(当前时刻)的收盘价的log_return.
        该horizon_log_retrun 不能作为特征列,仅作为Finance Environment中,设计reward function使用;
        并且,horizon_log_return最后horizon几个时刻值是NaN,并为作为异常值删去,这是因为这几个horizon时刻,包含各个训练用features的state是存在合理值.
        1.
        只观察股票数据中的收盘价,并将收盘价做以下处理,生成各个特征:
        log_return: np.log(df/df.shift()) 相邻时序收盘价格比值,取对数,反映收益;
        Roll_price_sma: 收盘价Rolling指定window后,对window内取mean,即: simple moving average;
        Roll_price_min: 收盘价Rolling指定window后,对window内取min,即: 滚动最小值;
        Roll_price_max: 收盘价Rolling指定window后,对window内取max,即: 滚动最大值;
        Roll_return_momt: log_return的Rolling指定window后,对window内取mean,即: 动量;
        Roll_return_std: log_return的Rolling指定window后,对window内取std,即: 滚动波动率;
        (Mark_return : log_return正值取1,负值取0;即:收益正负标记; 已取消)
        以上6列,再每列滑动lags时序,生成lags*7列,列名: lag1_columnname.
        2.
        添加与市场相关的MarketFactor,包括:
        除基础的指标: ['d_amplitude', 'volume', 'RSI14', 'high', 'low', 'open','close']之外,
        新增加指标有:
            # log{Today's Close Price - (T-5) day's close price}; (5:weekday)
            # log{high/low};
            # log{Today's Open Price - (T-5) day's open price};
            # Close/Previous Close;
            # Open/Previous Close;
            # High/Previous Close;
            # Low/Previous Close
        3.
        最后1列,为收盘价列;列名:close_colName;

        Args:
            by_BinanceAPI : 是否使用BinanceAPI通道,True时走BinanceAPI;False时,走CoinCapAPI,或其它header类似的API.
            FromWeb: 是否从网站上爬取(True);或者从存储的json文件中读取;
            horizon: 当前t时刻之后的t_horizon时刻,用于预测或者关注reward的t_horizon时刻与t时刻的log_return;
            interval:  (str) – the interval of kline, e.g 1m, 5m, 1h, 1d, etc. (仅适用于BinanceAPI)
                        '12h'代表12小时的数据,日24小时,则每两个序列表示一天;'6h'代表6小时的数据,则每4个序列表示一天;
            DayH: H12代表12小时的数据,日24小时,则每两个序列表示一天;DayH=24/12;新版弃而不用,改为通过interval自动计算;
            MarketFactor: 是否输出新增加的Market_Factor指标;Market_Factor=False时,仅输出close单列运算演化的列;
            weekdays: 一周有几天;周末休市是5天;全天候是7天;

            (股票数据DataFrame,包含有收盘价的列,列名为:'close_colName' (例如:'close');)
            close_colName: 股票数据中收盘价的列名;
            lags: 延时滑动的时序数;
            window: Rolling的window大小

        """
        if by_BinanceAPI:
            data = self.BinanceAPI_2_DF(FromWeb=FromWeb, interval=interval)  # 默认interval= '12h'
        else:
            data = self.GenDF_Frjson_FrWeb(FromWeb=FromWeb)
        # 1 只观察股票数据中的收盘价,并将收盘价做以下处理,生成各个特征:
        # df = pd.DataFrame(data[close_colName])  # 原DataFrame中,除收盘价以外的其它列舍弃;
        # df.dropna(inplace=True)
        data["log_return"] = np.log(
            data[close_colName] / data[close_colName].shift())  # 收益率(对数)
        data["Roll_price_sma"] = (
            data[close_colName].rolling(window).mean()
        )  # simple moving average
        data["Roll_price_min"] = (
            data[close_colName].rolling(window).min()
        )  # 滚动最小值与滚动最大值、动量以及滚动波动率
        data["Roll_price_max"] = data[close_colName].rolling(window).max()  # 滚动最大值;
        data["Roll_return_mom"] = data["log_return"].rolling(window).mean()  # 动量;
        data["Roll_return_std"] = data["log_return"].rolling(window).std()  # 滚动波动率
        data.dropna(inplace=True)
        # data["Mark_return"] = np.where(data["log_return"] > 0, 1, 0)
        features = [
            "log_return",
            "Roll_price_sma",
            "Roll_price_min",
            "Roll_price_max",
            "Roll_return_mom",
            "Roll_return_std",
            # "Mark_return",
            close_colName,
        ]
        features_afterLags = features[:]  # 实现浅copy,分配不同的内存,否则features.append陷入死循环
        for f in features:
            for lag in range(1, lags + 1):  # 收盘价列不再滑动延时;
                col = "{}_lag{}".format(f, lag)
                features_afterLags.append(col)
                data[col] = data[f].shift(lag)
                data.dropna(inplace=True)

        # data["log_return_unnormalized"] = data["log_return"] # 复制该列,列入non_state_features,不参与normalization;

        # 2 添加与市场相关的MarketFactor,
        if MarketFactor:
            # # 日振幅: 特征太多,取消
            # data['d_amplitude'] = data['high'] - data['low']

            # log{Today's Close Price - (T-5) day's close price};
            data['Log_close_weeklag'] = np.log(
                data[close_colName] / data[close_colName].shift(periods=weekdays))
            # log{high/low};
            data['Log_high_low'] = np.log(data['high'] / data['low'])
            # log{Today's Open Price - (T-5) day's open price};
            data['Log_open_weeklag'] = np.log(
                data['open'] / data['open'].shift(periods=weekdays))

            # Open/Previous Close;
            data['open_pre_close'] = data['open'] / data[close_colName].shift()
            # High/Previous Close;
            data['high_pre_close'] = data['high'] / data[close_colName].shift()
            # Low/Previous Close
            data['low_pre_close'] = data['low'] / data[close_colName].shift()

            # H12代表12小时的数据,24小时,则每两个序列表示一天
            # DayH = 24/12
            interval_unit = int(interval[:-1])  # interval字符串,例如: 12h
            time_unit = interval[-1]
            if time_unit == 'h':
                DayScale = 24 / interval_unit
            elif time_unit == 'd':
                DayScale = 24 / (interval_unit * 24)
            elif time_unit == 'm':
                DayScale = 24 / (interval_unit / 60)
            else:
                print(f"the interval({interval}) you input is with Invalid Time Unit")
                DayScale = 'invalid time unit'

            RSI_period = int(weekdays * DayScale)
            RSI = self.RSI(data[close_colName], periods=RSI_period)
            RSI_columnName = 'RSI_' + str(weekdays)
            data[RSI_columnName] = RSI

            data.dropna(inplace=True)

            features_afterLags.remove(close_colName)  # close列计划放入最后1列,所以这里先删除

            # 交易数据的最后horizon个时刻,假设交易数据最后的时刻为T,则(T - horizon, T)的horizon段内的horizon_price列数据为:NaN;
            # NaN仍然保留,并未删去; 因为(T-horizon,T)的state是存在的;也基于同样的原因,该列放在所有dropna之后,并且不能作为features进入训练;
            # 这里使用price而不是log_return,是因为后面奖励函数需要扣除交易手续费,交易手续费是基于price;
            # data["horizon_log_return"] = (data[close_colName].shift(-horizon) / data[close_colName])  # 收益率(对数)
            data["horizon_price"] = data[close_colName].shift(-horizon)

            # BinanceAPI通道来的数据,有列: [open,high,low,close,volume,amount,num_trades,bid_volume,bid_amount]
            # volume:成交量(单位为手);amount:成交量(单位为金额),特征缩减,amount取消;同理,bid_amount取消;
            if by_BinanceAPI:
                X = data[features_afterLags + ['volume', RSI_columnName, 'Log_close_weeklag', 'Log_high_low',
                                               'Log_open_weeklag', 'open_pre_close', 'high_pre_close',
                                               'low_pre_close', 'num_trades', 'bid_volume',
                                               'horizon_price', close_colName]]
            else:
                X = data[features_afterLags + ['volume', RSI_columnName, 'Log_close_weeklag', 'Log_high_low',
                                               'Log_open_weeklag', 'open_pre_close', 'high_pre_close', 'low_pre_close',
                                               'horizon_price', close_colName]]
        else:

            # 交易数据的最后horizon个时刻,假设交易数据最后的时刻为T,则(T - horizon, T)的horizon段内的horizon_price列数据为:NaN;
            # NaN仍然保留,并未删去; 因为(T-horizon,T)的state是存在的;也基于同样的原因,该列放在所有dropna之后,并且不能作为features进入训练;
            # data["horizon_log_return"] = np.log(data[close_colName].shift(-horizon) / data[close_colName])  # 收益率(对数)
            data["horizon_price"] = data[close_colName].shift(-horizon)

            features_afterLags.remove(close_colName)  # close列计划放入最后1列,所以这里先删除
            X = data[features_afterLags + ["horizon_price"] + [close_colName]]

        display(X.describe())
        display(X.head(2))
        display(X.tail(2))
        return X
