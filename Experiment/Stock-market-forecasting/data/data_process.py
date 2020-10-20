import os, sys
import pandas as pd
from datetime import datetime
import numpy as np
import json


def missData():
    keys = ["{}-{:02d}".format(year, month) for year in range(1990, 2019) for month in range(1, 13)]
    data_dict = dict.fromkeys(keys)
    path = './exceptions.txt'
    # 字典
    for line in open(path):
        ticker = line.split(" ")[0]
        date = line.split(" ")[2]
        if(data_dict[date] == None):
            data_dict[date] = [ticker]
        elif isinstance(data_dict[date], list):
            data_dict[date].append(ticker)
    # 写入json文件
    jso_str = json.dumps(data_dict, indent=4, ensure_ascii=False)
    with open("./exceptions.josn", "w", encoding='utf-8') as json_file:
        json_file.write(jso_str)