import quandl
import os
import pandas as pd
from datetime import datetime
import numpy as np
import calendar


file_exceptions = "exceptions.txt"
tickers = pd.read_csv('SPXconst.csv')


API_KEY = '95NJZYJYyXToregziQw8'

# 获取股票的开盘价和收尾价
def getInfo(ticker, start_date, end_date):
    data = quandl.get("WIKI/{}".format(ticker.replace(".","_")), start_date=start_date, end_date=end_date, api_key=API_KEY)
    data = data.loc[:, ['Open', 'Close']]
    return data['Open'], data['Close']


years = [str(x) for x in range(1990, 2019)]

for year in years:
    tickers_open = pd.DataFrame()
    tickers_close = pd.DataFrame()

    months = [("{:02d}".format(x) + '/' + year) for x in range(1, 13)]
    for month in months:
        # 获取日期信息，以及所有的股票名称
        month_str = month
        month = datetime.strptime(month, "%m/%Y") # 转换成时间戳
        monthFirstDay = datetime(month.year, month.month, 1).strftime(('%Y-%m-%d'))
        monthLastDay = datetime(month.year, month.month, calendar.monthrange(month.year, month.month)[1]).strftime(('%Y-%m-%d'))
        tickers_month =tickers[month_str].dropna().tolist()

        # 获得每个月每支股票的信息， index为日期， cols为股票名称
        month_open = pd.DataFrame()
        month_close = pd.DataFrame()
        for ticker in tickers_month:
            try:
                data_open, data_close = getInfo(ticker, monthFirstDay, monthLastDay)
                # 补NaN
                month_open = pd.concat([month_open, data_open], join='outer', axis=1)
                month_close = pd.concat([month_close, data_close], join='outer', axis=1)
                print("{} in {} finished!".format(ticker, month_str))
            except:
                with open(file_exceptions, 'a') as f:
                    f.write("{} in {} not exist!\n".format(ticker, month_str))
                print("{} in {} not exist!".format(ticker, month_str))
                
        print("Infomation in {} finished!".format(month_str))
        # 将一年中所有月份的股票信息合并， 空缺处为NaN 
        tickers_open = pd.concat([tickers_open, month_open])
        tickers_close = pd.concat([tickers_close, month_close])

    # 存储每年的股票信息
    tickers_open.to_csv("Open-{}.csv".format(year))
    tickers_close.to_csv("Close-{}.csv".format(year))
    print("S&P500 price data in {} finished!".format(year))
