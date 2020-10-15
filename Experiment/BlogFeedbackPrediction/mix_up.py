import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn import preprocessing
from sklearn.ensemble import BaggingRegressor
import math, os, random 
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


# Plot the learning curve
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
  
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5), squeeze=False)

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=make_scorer(r2_score))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    print("the shape of train_scores_mean", train_scores_mean.shape)
    print(train_scores_mean)
    print("the shape of test_scores_mean", test_scores_mean.shape)
    print(test_scores_mean)
    
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training r2_score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Testing r2_score")
    axes[0].legend(loc="best")

    return plt

def mixup_data(x, y, alpha=0.2, use_cuda=False):

    # pandas DataFrame to numpy array
    x = x.values
    y = y.values

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    train_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = np.random.permutation(train_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_y = lam * y + (1 - lam) * y[index]

    # numpy array to pandas DataFrame
    mixed_x = pd.DataFrame(mixed_x)
    mixed_y = pd.Series(mixed_y)

    return mixed_x, mixed_y


# Load the train data and test data

def load_data():
    files = os.listdir("Data")
    train_file = "blogData_train.csv"
    
    train_data = pd.read_csv("./Data/{}".format(train_file),header=None)
    train_output = train_data[len(train_data.columns)-1]
    train_num = train_data.shape[0]
    del train_data[len(train_data.columns)-1]

    files.remove(train_file)
    file_list = files
    test_data = pd.DataFrame()
    for filename in file_list:
        df = pd.read_csv("./Data/{}".format(filename),header=None)
        test_data = pd.concat([test_data, df], axis=0)
    test_output = test_data[len(test_data.columns)-1]
    del test_data[len(test_data.columns)-1]

    data_X = pd.concat([train_data, test_data], axis=0)
    data_Y = pd.concat([train_output, test_output], axis=0)

    return data_X, data_Y, train_num


def load_data_mixup():
    files = os.listdir("Data")
    train_file = "blogData_train.csv"
    
    train_data = pd.read_csv("./Data/{}".format(train_file),header=None)
    train_output = train_data[len(train_data.columns)-1]
    train_num = train_data.shape[0]
    del train_data[len(train_data.columns)-1]
    # mix up
    train_data , train_output = mixup_data(train_data, train_output)

    files.remove(train_file)
    file_list = files
    test_data = pd.DataFrame()
    for filename in file_list:
        df = pd.read_csv("./Data/{}".format(filename),header=None)
        test_data = pd.concat([test_data, df], axis=0)
    test_output = test_data[len(test_data.columns)-1]
    del test_data[len(test_data.columns)-1]

    
    data_X = pd.concat([train_data, test_data], axis=0)
    data_Y = pd.concat([train_output, test_output], axis=0)

    return data_X, data_Y, train_num

def train():
    fig, axes = plt.subplots(1, 1, figsize=(10, 15), squeeze=False)
    
    data_X, data_Y, train_num = load_data()
    # print(data_Y.shape, data_Y.shape)
    # print(train_num, data_X.shape[0])
    
    train_indices = [list(range(0, train_num))]
    test_indices =  [list(range(train_num, data_X.shape[0]))]
    custom_cv = zip(train_indices, test_indices)
    
    rf = RandomForestRegressor(n_estimators=100, max_features=100)
    plot_learning_curve(estimator=rf, title="Random Forest", X=data_X, y=data_Y, axes=axes[:,0], cv=custom_cv, train_sizes=np.linspace(.1, 0.87, 100))
    # plt.show()
    plt.savefig("./learning_curve(rf).png")
    plt.close()

def train_mixup():
    print("Mix Up")
    fig, axes = plt.subplots(1, 1, figsize=(10, 15), squeeze=False)
    
    data_X, data_Y, train_num = load_data_mixup()
    print(data_X.shape, data_Y.shape)
    print(train_num, data_X.shape[0])
    train_indices = [list(range(0, train_num))]
    test_indices =  [list(range(train_num, data_X.shape[0]))]
    custom_cv = zip(train_indices, test_indices)
    
    rf = RandomForestRegressor(n_estimators=100, max_features=100)
    plot_learning_curve(estimator=rf, title="Random Forest", X=data_X, y=data_Y, axes=axes[:,0], cv=custom_cv, train_sizes=np.linspace(.1, 0.87, 100))
    # plt.show()
    plt.savefig("./learning_curve(rf mix_up).png")
    plt.close()


def main():
    # train()
    train_mixup()


if __name__ == "__main__":
    main()