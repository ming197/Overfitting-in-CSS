import quandl
import os, sys
import pandas as pd
from datetime import datetime
import numpy as np
import calendar
import time
import threading



path = "./data_csv"
file_exceptions = "exceptions.txt"
tickers = pd.read_csv('SPXconst.csv')


API_KEY = ['95NJZYJYyXToregziQw8', 'yJAooAzza7L4EJxyFkcs', '2H5CLCnZE_AUSAzUhicw', 'KhALVVzAJx-FK-XsFxvS', 'jMVcgK1aP-MUmyKChg9-']
api_index = 0
calls_limit = 50000
calls_num = [0 for x in API_KEY]
calls_flag = [False for x in API_KEY]
shutdown = False

class MyThread(threading.Thread):
    def __init__(self, fun, args):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.result = None
        self.error = None
        self.fun = fun
        self.args = args

    def run(self):
        try:
            self.result = self.fun(self.args[0], self.args[1], self.args[2])
        except:
            self.error = sys.exc_info()

# 获取股票的开盘价和收尾价
def getInfo(ticker, start_date, end_date):
    # 声明全局变量
    global api_index, shutdown
    # API调用数据
    for i in range(len(API_KEY)):
        if calls_flag[api_index] == True:
            api_index = (api_index + 1) % len(API_KEY)
        else:
            break
        shutdown = (shutdown and calls_flag[api_index])

    if(shutdown == True):
        return None

    calls_num[api_index] += 1
    if(calls_num[api_index] == calls_limit):
        calls_flag[api_index] = True

    data = None
    data = quandl.get("WIKI/{}".format(ticker.replace(".","_")), start_date=start_date, end_date=end_date, api_key=API_KEY[api_index])
    data = data.loc[:, ['Open', 'Close']]
    api_index = (api_index + 1) % len(API_KEY)
    time.sleep(2)

    return data


def main():
    
    years = [str(x) for x in range(1990, 2019)]
    for year in years:
        months = [("{:02d}".format(x) + '/' + year) for x in range(1, 13)]
        for month in months:
            # 获取日期信息，以及所有的股票名称
            month_str = month
            month = datetime.strptime(month, "%m/%Y") # 转换成时间戳
            monthFirstDay = datetime(month.year, month.month, 1).strftime(('%Y-%m-%d'))
            monthLastDay = datetime(month.year, month.month, calendar.monthrange(month.year, month.month)[1]).strftime(('%Y-%m-%d'))
            tickers_month = tickers[month_str].dropna().tolist()
            month_str = '-'.join(month_str.split('/')[::-1])

            for ticker in tickers_month:
                runtimes = 0
                flag = False
                while(runtimes < len(API_KEY) and not flag):
                    # 定义同步子线程，并设置时间限制
                    args = [ticker, monthFirstDay, monthLastDay]
                    t = MyThread(fun=getInfo, args=args)
                    t.start()
                    t.join(60)
                    if(shutdown == True):
                        return
                    # 抛出异常，该ticker的数据不存在，跳出循环
                    if(t.error != None):
                        with open(file_exceptions, 'a') as f:
                            f.write("{} in {} not exist!\n".format(ticker, month_str))
                        print("{} in {} not exist!".format(ticker, month_str))
                        break
                    # 存储返回值
                    data = t.result
                    if isinstance(data, pd.DataFrame):
                        flag = True
                        # 将数据写入csv文件
                        data.to_csv("{}/{} {}.csv".format(path, month_str, ticker))
                        print("{} in {} finished!".format(ticker, month_str))
                    elif data == None:
                        runtimes += 1
                            
                
                    
                
            print("Infomation in {} finished!".format(month_str))
        

            
       
if __name__ == "__main__":
    main()
    if(shutdown):
        print("call nums reach limits!")
    
    
    