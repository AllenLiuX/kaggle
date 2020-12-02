# -- coding:UTF-8 --
import numpy as np
import pandas as pd
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import modules as md
import time
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# from fbprophet import Prophet
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
# import torch.nn.functional as F
# from sklearn.metrics import mean_squared_error


def main():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv", )
    real_df = pd.read_csv("data/test-real.csv")
    features = train_df.columns.ravel()
    states = list(set(train_df['Province_State'].values))

    # data pre-processing
    state_to_traindf = {}
    state_to_realdf = {}
    for state in states[:2]:
        # train dfs
        train_state_df = train_df[train_df['Province_State'] == state]
        train_state_df = train_state_df[['Date', 'Confirmed', 'Deaths']]
        days = train_state_df['Date'].apply(md.date2days)
        train_state_df['Date'] = days
        state_to_traindf[state] = train_state_df
        # real dfs
        real_state_df = real_df[train_df['Province_State'] == state]
        real_state_df = real_state_df[['Date', 'Confirmed', 'Deaths']]
        days = real_state_df['Date'].apply(md.date2days)
        real_state_df['Date'] = days
        state_to_realdf[state] = real_state_df
    # print(state_to_traindf['California'])
    # print(state_to_realdf['California'])

    # training
    all_pred_df = pd.DataFrame(columns=['Province_State', 'Date', 'Confirmed', 'Deaths'])
    for state in states[:2]:
        # TODO replace training function here:
        pred_df = nonlinear_regression(state_to_traindf[state], state_to_realdf[state])
        # pred_df = lstm(state_to_traindf[state], state_to_realdf[state])

        pred_df['Province_State'] = state
        all_pred_df = pd.concat([all_pred_df, pred_df], ignore_index=True)

    all_pred_df['ForcastID'] = all_pred_df.apply(lambda row: md.get_forcast_id(row['Province_State'], row['Date'], test_df), axis=1)
    print(all_pred_df)



    print(md.df_MAPE(test_df, real_df))

'''
input are two df that has three columns: Date(number of days), Confirmed, and Deaths
output is one df that has three columns: Date(real date begins from 09-01-2020), Confirmed, and Deaths
'''
def nonlinear_regression(train_df, real_df):
    model = Sequential()
    model.add(Dense(units=150, input_dim=1))  # 输入维度和输出维度
    model.add(Activation('relu'))
    # model.add(Dense(units=20))  # 输入维度和输出维度
    # model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.add(Activation('relu'))
    adam = Adam(lr=0.3)
    sgd = SGD(lr=0.2)
    model.compile(optimizer=adam, loss='mse')

    pred = {}
    for feature in ['Confirmed', 'Deaths']:
        x_data = train_df['Date']
        y_data = train_df[feature]
        for step in range(4001):
            cost = model.train_on_batch(x_data, y_data)
            if step % 500 == 0:
                print('cost: ', cost)
        x_real = real_df['Date']
        y_real = real_df[feature]
        y_pred = model.predict(x_data.append(x_real))
        plt.scatter(x_data, y_data, s=10)
        plt.scatter(x_real, y_real, s=10)
        plt.plot(x_data.append(x_real), y_pred, 'r-')
        plt.title(feature)
        plt.show()
        # format predictions into df
        y_pred = model.predict(x_real)
        y_pred = y_pred.reshape(1, -1)[0]
        pred.update({
            'Date': [md.days2date(i) for i in x_real],
            feature: y_pred
        })
    pred_df = pd.DataFrame(pred)
    print(pred_df)
    return pred_df


def lstm(train_df, real_df):
    epochs = 20
    batch_size = 16
    time_stamp = 2
    pred = {}
    for feature in ['Confirmed', 'Deaths']:
        x_data = train_df['Date']
        y_data = train_df[feature]
        x_train = []
        y_train = []
        for i in range(time_stamp, len(x_data)):
            x_train.append([x_data[i - time_stamp:i], y_data[i - time_stamp:i]])
            y_train.append(y_data.values[i])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # print(x_train)
        # print(y_train)
        # print(x_train.shape[-1], x_train.shape[1])
        model = Sequential()
        model.add(LSTM(units=100, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
        model.add(Dense(50))
        model.add(Dense(1))
        adam = Adam(lr=0.2)
        model.compile(loss='mean_squared_error', optimizer=adam)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        x_real = real_df['Date']
        y_real = real_df[feature]

        plot = False
        if plot:
            # x_pred = []
            for i in range(time_stamp, len(x_real)):
                x_train = np.append(x_train, [x_real[i - time_stamp:i], y_real[i - time_stamp:i]])
            x_train = np.array(x_train)
            # print(x_pred)
            y_pred = model.predict(x_train)
            y_pred = y_pred.reshape(1, -1)[0]
            plt.scatter(x_data, y_data, s=10)
            plt.scatter(x_real, y_real, s=10)
            plt.plot(x_data.append(x_real), y_pred, 'r-')
            plt.title(feature)
            plt.show()

        x_pred = []
        for i in range(time_stamp, len(x_real)):
            x_pred.append([x_real[i - time_stamp:i]])
        x_pred = np.array(x_pred)
        y_pred = model.predict(x_pred)
        y_pred = y_pred.reshape(1, -1)[0].tolist()
        y_pred += [y_pred[-1], y_pred[-1]]
        print(len(x_real), len(y_pred))
        pred.update({
            'Date': [md.days2date(i) for i in x_real],
            feature: y_pred
        })
    print(pred)
    pred_df = pd.DataFrame(pred)
    print(pred_df)
    return pred_df


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('======= Time taken: %f =======' %(end_time - start_time))
