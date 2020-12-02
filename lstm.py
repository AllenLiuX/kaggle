import numpy as np
import pandas as pd
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import modules as md

import seaborn as sns
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib import ticker
time_stamp = 10

traindf = pd.read_csv("data/train.csv")
testdf = pd.read_csv("data/test-real.csv", )
# mobility = pd.read_csv("data/graph.csv")
print(traindf.shape)
features = traindf.columns.ravel()
print(features)


class Predicator():
    def __init__(self, state):
        self.state_df = traindf[traindf['Province_State'] == state]
        self.state_df = self.state_df[['Date', 'Confirmed', 'Deaths']]
        days = self.state_df['Date'].apply(md.date2days)
        self.state_df['Date'] = days


    def preparation(self):
        split = int(0.9 * self.state_df.shape[0])
        self.train = self.state[0:split]
        self.valid = self.state_df[split:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(train)



# TODO data pre-processing
cal_df = traindf[traindf['Province_State'] == 'California']
cal_df = cal_df[['Date', 'Confirmed', 'Deaths']]

days = cal_df['Date'].apply(md.date2days)
cal_df['Date'] = days
print(cal_df)

split = int(0.9*cal_df.shape[0])
train = cal_df[0:split]
valid = cal_df[split:]

print(train.shape, valid.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)

# print(scaled_data)
x_train, y_train = [], []

# 训练集
for i in range(time_stamp, len(train)):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i, 2])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train, y_train)

# # 验证集
# scaled_data = scaler.fit_transform(valid)
# x_valid, y_valid = [], []
# for i in range(time_stamp, len(valid)):
#     x_valid.append(scaled_data[i - time_stamp:i])
#     y_valid.append(scaled_data[i, 2])
#
# x_valid, y_valid = np.array(x_valid), np.array(y_valid)



test_cal_df = testdf[testdf['Province_State'] == 'California'][['Date', 'Confirmed', 'Deaths']]
days = test_cal_df['Date'].apply(md.date2days)
test_cal_df['Date'] = days
scaled_test = scaler.transform(test_cal_df)
valid = test_cal_df
x_valid, y_valid = [], []
for i in range(time_stamp, len(scaled_test)):
    x_valid.append(scaled_test[i - time_stamp:i])
    y_valid.append(scaled_test[i, 2])

x_valid, y_valid = np.array(x_valid), np.array(y_valid)

# print(test_cal_df)

# # TODO Model

epochs = 15
batch_size = 16

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_dim= x_train.shape[-1], input_length=x_train.shape[1]))
model.add(LSTM(units=50))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
#
# # TODO Forcast
prediction = model.predict(x_valid)
scaler.fit_transform(pd.DataFrame(valid['Deaths'].values))
# 反归一化
prediction = scaler.inverse_transform(prediction)
y_valid = scaler.inverse_transform([y_valid])
# print(y_valid)
# print(prediction)
rms = np.sqrt(np.mean(np.power((y_valid - prediction), 2)))
print(rms)
print(prediction)
print(y_valid)

plt.figure(figsize=(16, 8))
dict_data = {
    'Predictions': prediction.reshape(1,-1)[0],
    'Close': y_valid[0]
}
data_pd = pd.DataFrame(dict_data)

plt.plot(data_pd[['Close', 'Predictions']])
plt.show()




# # 数据长度 一行28像素
# input_size = 28
# # 序列长度 28行
# time_steps = 28
# # 隐藏层cell个数
# cell_size = 50

# f = np.load('mnist.npz')
# x_train, y_train = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()
# # (x_train, y_train), (x_test, y_test) = np.load('mnist.npz')
# print('x_shape:', x_train.shape) #60000
# print('y_shape:', y_train.shape) #60000
# # (60000, 28, 28)
#
# x_train = x_train/255.0   #/255 -> 均一化, -1表示自动计算行数
# x_test = x_test/255.0
#
# #换one hot格式
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
#
# # 创建模型
# model = Sequential()
# model.add(SimpleRNN(
#     units=cell_size,  # output
#     input_shape=(time_steps, input_size)  # input
# ))
# # 输出层
# model.add(Dense(10, activation='softmax'))
#
# # 优化器
# adam = Adam(lr=1e-4)
#
# # 定义优化器，loss，训练过程中计算准确率
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 训练模型
# model.fit(x_train, y_train, batch_size=64, epochs=10)
#
# # 评估模型
# loss, accuracy = model.evaluate(x_test, y_test)
# print('\ntest loss', loss)
# print('test accuracy', accuracy)
#
# loss, accuracy = model.evaluate(x_train, y_train)
# print('\ntrain loss', loss)
# print('train accuracy', accuracy)
#
# # 保存模型
# model.save('model.h5')
#
# # 存参数，取参数
# model.save_weights('model_weights.h5')
# model.load_weights('model_weights.h5')
#
# # 保存/载入网络结构
# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
