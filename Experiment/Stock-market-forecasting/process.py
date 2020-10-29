import pandas as pd 
import numpy as np
import os

path = "./data"
files = os.listdir(path)
open_file = []
close_flie = []
for file_item in files:
    if file_item[0:4] == "Open":
        open_file.append(file_item)
    elif file_item[0:5] == "Close":
        close_flie.append(file_item)
sorted(open_file)
sorted(close_flie)
for i in range(len(open_file) - 3):
    # Open 
    df = pd.read_csv("{}/{}".format(path,open_file[i]),index_col="Date")
    df1 = pd.read_csv("{}/{}".format(path,open_file[i + 1]),index_col="Date")
    df2 = pd.read_csv("{}/{}".format(path,open_file[i + 2]),index_col="Date")
    df3 = pd.read_csv("{}/{}".format(path,open_file[i + 3]),index_col="Date")
    df = pd.concat([df, df1, df2, df3])
    df.sort_index(inplace=True)
    df.index.name = "Date"
    df.to_csv("{}/{}".format(path,open_file[i]))
    # Close
    df = pd.read_csv("{}/{}".format(path,close_flie[i]),index_col="Date")
    df1 = pd.read_csv("{}/{}".format(path,close_flie[i + 1]),index_col="Date")
    df2 = pd.read_csv("{}/{}".format(path,close_flie[i + 2]),index_col="Date")
    df3 = pd.read_csv("{}/{}".format(path,close_flie[i + 3]),index_col="Date")
    df = pd.concat([df, df1, df2, df3])
    df.sort_index(inplace=True)
    df.index.name = "Date"
    df.to_csv("{}/{}".format(path,close_flie[i]))
    print("{} {} Finished!".format(open_file[i], close_flie[i]))
    
print("Finished!")