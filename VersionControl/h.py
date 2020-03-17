import numpy as np
import pandas as pd
import datetime
# np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from VersionControl import version_contrl as vc

# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 获取数据

# data = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
# data['result'] = y_train
# data1 = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
# data1['result'] = y_test
# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # 标准化
X_test = X_test.reshape(X_test.shape[0], -1) / 255.
# 转换为one_hot形式
# keras_v1 y_train = np_utils.to_categorical(y_train, nb_classes=10)
# keras_v1 y_test = np_utils.to_categorical(y_test, nb_classes=10)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(y_test.shape)


# data = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
# data['result'] = y_train
# data1 = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))
# data1['result'] = y_test
# data.to_csv('mnist_train.csv', index=False)
# data1.to_csv('mnist_test.csv', index=False)


# 构建模型阶段
# 一次性搭建神经网络模型的方法，注意是将模型元素放在一个list中
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# 激活模型
# 自定义RMS优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# 训练阶段

# print('Training ------------')
# start = datetime.datetime.now()
# for i in range(10):
#     print(i,'*'*40)
#     a = 6000*i
#     b = 6000*(i+1)
#     model.fit(X_train[a:b], y_train[a:b], epochs=5, batch_size=30)  # keras_v1 epochs 2 nb_epoch
#     print('\nTesting ------------')
#     loss, accuracy = model.evaluate(X_test, y_test)
#     print('test loss: ', loss)
#     print('test accuracy: ', accuracy)
#     if accuracy > 0.95:
#         break
# end = datetime.datetime.now()
# print(end - start)




start = datetime.datetime.now()
print('Training ------------')
model.fit(X_train[0:30000], y_train[0:30000], epochs=1, batch_size=32)  # keras_v1 epochs 2 nb_epoch
end = datetime.datetime.now()
print(model.predict(X_test[0:1]))
# 测试阶段
print('\nTesting ------------')
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
print(end - start)
# # a = vc()
# # ss = data.columns.values.tolist()
# # a.save_version(model, data.values, data1.values, ss)



