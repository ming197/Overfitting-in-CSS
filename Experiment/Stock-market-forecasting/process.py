import pandas as pd 
import numpy as np
import os

path = "./data"

files = os.listdir(path)

for file in files:
    if file[0:4] != "Open" or files[0:4] != "Close":
        continue

    df = pd.read_csv("{}/{}".format(path, file), index_col="Date")
    df = df.sort_index()
    df.index.name = "Date"
    df.to_csv("{}/{}".format(path, file), index=None)