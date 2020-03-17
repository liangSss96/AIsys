# import pandas as pd
# import numpy as np
# a = pd.DataFrame([[[1,2]],
#                   [[3,4]]], columns=['1'])
# print(a['1'].apply(lambda x: x[1]))
# a.to_csv('eq.csv')
# a = pd.read_csv('eq.csv')
# print(type(a))
# print(type(a['1'].iloc[0]))
# b = a['1'].apply(lambda x: x[1:-1].split(','))
# print(type(b[0]))


# from VersionControl import version_contrl as vc
# a = vc()
# a.drawall()

# a = '[1,2,3]'
# print(a)
# print(a[1:-1].split(','))

# a = pd.DataFrame(np.arange(16).reshape(4,-1), columns=list('abcd'))
# a.to_csv('ad.csv')
# a = pd.read_csv('ad.csv')
# print(type(a['d']))
# b = a['d'].apply(lambda x: x+1)
# print(b)
# print(type(b))

# def strtolist(data):
#     str = data[1:-1]
#     l = str.split(',')
#     for i in range(len(l)):
#         l[i] = float(l[i])
#     return l
#
# str = '[1,2,3]'
# print(strtolist(str))
from VersionControl import version_contrl as vc
import numpy as np
from keras.datasets import boston_housing
from keras import models
from keras import layers
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
# print(X_train,y_train)
#加载数据 #对数据进行标准化预处理，方便神经网络更好的学习
mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std
x_train = pd.DataFrame(X_train)
x_test = pd.DataFrame(X_test)
x_train['true'] = y_train
x_test['true'] = y_test
# print(x_test.values.shape)
# print(X_test.shape)

ss = x_train.columns.values.tolist()

def build_model():
    model = models.Sequential()
    #进行层的搭建，注意第二层往后没有输入形状(input_shape)，它可以自动推导出输入的形状等于上一层输出的形状
    model.add(layers.Dense(64, activation='relu',input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #编译网络
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

num_epochs = 100
model = build_model()
# f = 'a.txt'
# with open(f, 'a') as file:
#     # Pass the file handle in as a lambda function to make it callable
#     model.summary(print_fn=lambda x: file.write(x + '\n'))
#     for i in range(len(model.layers)):
#         file.write(str(model.get_layer(index=i).get_config()) + '\n')
model.fit(X_train, y_train,epochs=num_epochs, batch_size=1, verbose=0)
# predicts = np.squeeze(model.predict(X_test))
# print(np.mean((predicts - y_test) ** 2))
a = vc()
a.save_version(model, x_train.values, x_test.values, ss)