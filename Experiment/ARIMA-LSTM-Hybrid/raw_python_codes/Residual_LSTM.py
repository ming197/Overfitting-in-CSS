import pandas as pd
import numpy as np
import os, random
import argparse
import ast
from tensorflow import set_random_seed
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 固定随机数种子
SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

parser = argparse.ArgumentParser(description='Parameters Configuration!')
parser.add_argument('--reg', '-r', help='正则化', type=ast.literal_eval, default=False)
parser.add_argument('--dropout', '-d', help='Dropout', type=ast.literal_eval, default=False)
parser.add_argument('--epochs', '-e', help='Dropout', type=ast.literal_eval, default=350)
args = parser.parse_args()
Reg = args.reg
Dropout = args.dropout
epochs = args.epochs


d = 'hybrid_LSTM'


# Train - Dev - Test Generation
train_X= pd.read_csv('../train_dev_test/after_arima/train_X.csv')
print('loaded train_X')
dev_X = pd.read_csv('../train_dev_test/after_arima/dev_X.csv')
print('loaded dev_X')
test1_X = pd.read_csv('../train_dev_test/after_arima/test1_X.csv')
print('loaded test1_X')
test2_X = pd.read_csv('../train_dev_test/after_arima/test2_X.csv')
print('loaded test2_X')
train_Y = pd.read_csv('../train_dev_test/after_arima/train_Y.csv')
print('loaded train_Y')
dev_Y = pd.read_csv('../train_dev_test/after_arima/dev_Y.csv')
print('loaded dev_Y')
test1_Y = pd.read_csv('../train_dev_test/after_arima/test1_Y.csv')
print('loaded test1_Y')
test2_Y = pd.read_csv('../train_dev_test/after_arima/test2_Y.csv')
print('loaded test2_Y')
train_X = train_X.loc[:, ~train_X.columns.str.contains('^Unnamed')]
dev_X = dev_X.loc[:, ~dev_X.columns.str.contains('^Unnamed')]
test1_X = test1_X.loc[:, ~test1_X.columns.str.contains('^Unnamed')]
test2_X = test2_X.loc[:, ~test2_X.columns.str.contains('^Unnamed')]
train_Y = train_Y.loc[:, ~train_Y.columns.str.contains('^Unnamed')]
dev_Y = dev_Y.loc[:, ~dev_Y.columns.str.contains('^Unnamed')]
test1_Y = test1_Y.loc[:, ~test1_Y.columns.str.contains('^Unnamed')]
test2_Y = test2_Y.loc[:, ~test2_Y.columns.str.contains('^Unnamed')]

# data sampling
STEP = 20
#num_list = [STEP*i for i in range(int(1117500/STEP))]

_train_X = np.asarray(train_X).reshape((int(1117500/STEP), 20, 1))
_dev_X = np.asarray(dev_X).reshape((int(1117500/STEP), 20, 1))
_test1_X = np.asarray(test1_X).reshape((int(1117500/STEP), 20, 1))
_test2_X = np.asarray(test2_X).reshape((int(1117500/STEP), 20, 1))

_train_Y = np.asarray(train_Y).reshape(int(1117500/STEP), 1)
_dev_Y = np.asarray(dev_Y).reshape(int(1117500/STEP), 1)
_test1_Y = np.asarray(test1_Y).reshape(int(1117500/STEP), 1)
_test2_Y = np.asarray(test2_Y).reshape(int(1117500/STEP), 1)

#define custom activation
class Double_Tanh(Activation):
    def __init__(self, activation, **kwargs):
        super(Double_Tanh, self).__init__(activation, **kwargs)
        self.__name__ = 'double_tanh'

def double_tanh(x):
    return (K.tanh(x) * 2)

get_custom_objects().update({'double_tanh':Double_Tanh(double_tanh)})

# Model Generation
model = Sequential()
#check https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
print("Reg: ", Reg)
print("Dropout: ", Dropout)
if (not Reg and not Dropout):
    model.add(LSTM(25, input_shape=(20,1), dropout=0.0, kernel_regularizer=l1_l2(0.00,0.00), bias_regularizer=l1_l2(0.00,0.00)))
if (Reg and Dropout):
    d += '_with_reg_dropout'
    model.add(LSTM(25, input_shape=(20,1), dropout=0.2, kernel_regularizer=l1_l2(0.00,0.1), bias_regularizer=l1_l2(0.00,0.1)))
if (Reg and not Dropout):
    d += '_with_reg'
    model.add(LSTM(25, input_shape=(20,1), dropout=0.0, kernel_regularizer=l1_l2(0.00,0.1), bias_regularizer=l1_l2(0.00,0.1)))
if (Dropout and not Reg):
    d += '_with_dropout'
    model.add(LSTM(25, input_shape=(20,1), dropout=0.2, kernel_regularizer=l1_l2(0.00,0.00), bias_regularizer=l1_l2(0.00,0.00)))

model.add(Dense(1))
model.add(Activation(double_tanh))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])


print(model.metrics_names)
# Fitting the Model
model_scores = {}

epoch_num=1
for _ in range(epochs):

    # train the model
    dir = '../models/'+d
    # if dir not exit, make it
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Make the file: {}".format(dir))
    file_list = os.listdir(dir)

    if len(file_list) != 0 :
        epoch_num = len(file_list) + 1
        recent_model_name = 'epoch' + str(epoch_num-1) + '.h5'
        filepath = '../models/' + d + '/' + recent_model_name
        model = load_model(filepath)

    filepath = '../models/' + d + '/epoch' + str(epoch_num)+'.h5'

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
    callbacks_list = [checkpoint]
    if len(callbacks_list) == 0:
        model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True)
    else:
        model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True, callbacks=callbacks_list)

    # test the model
    score_train = model.evaluate(_train_X, _train_Y)
    score_dev = model.evaluate(_dev_X, _dev_Y)
    score_test1 = model.evaluate(_test1_X, _test1_Y)
    score_test2 = model.evaluate(_test2_X, _test2_Y)

    print('train set score : mse - ' + str(score_train[1]) +' / mae - ' + str(score_train[2]))
    print('dev set score : mse - ' + str(score_dev[1]) +' / mae - ' + str(score_dev[2]))
    print('test1 set score : mse - ' + str(score_test1[1]) +' / mae - ' + str(score_test1[2]))
    print('test2 set score : mse - ' + str(score_test2[1]) +' / mae - ' + str(score_test2[2]))
#.history['mean_squared_error'][0]

    # get former score data
    file_name = "../models/"+d+".csv"
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns = ["DEV_MAE","DEV_MSE","TEST1_MAE","TEST1_MSE","TEST2_MAE","TEST2_MSE","TRAIN_MAE","TRAIN_MSE"])
    else:
        df = pd.read_csv(file_name)
        
    train_mse = list(df['TRAIN_MSE'])
    dev_mse = list(df['DEV_MSE'])
    test1_mse = list(df['TEST1_MSE'])
    test2_mse = list(df['TEST2_MSE'])

    train_mae = list(df['TRAIN_MAE'])
    dev_mae = list(df['DEV_MAE'])
    test1_mae = list(df['TEST1_MAE'])
    test2_mae = list(df['TEST2_MAE'])

    # append new data
    train_mse.append(score_train[1])
    dev_mse.append(score_dev[1])
    test1_mse.append(score_test1[1])
    test2_mse.append(score_test2[1])

    train_mae.append(score_train[2])
    dev_mae.append(score_dev[2])
    test1_mae.append(score_test1[2])
    test2_mae.append(score_test2[2])

    # organize newly created score dataset
    model_scores['TRAIN_MSE'] = train_mse
    model_scores['DEV_MSE'] = dev_mse
    model_scores['TEST1_MSE'] = test1_mse
    model_scores['TEST2_MSE'] = test2_mse

    model_scores['TRAIN_MAE'] = train_mae
    model_scores['DEV_MAE'] = dev_mae
    model_scores['TEST1_MAE'] = test1_mae
    model_scores['TEST2_MAE'] = test2_mae

    # save newly created score dataset
    model_scores_df = pd.DataFrame(model_scores)
    model_scores_df.to_csv("../models/"+d+".csv")

