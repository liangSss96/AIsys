from VersionControl import version_contrl as vc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error
import time
import pandas as pd
import warnings
import random
warnings.filterwarnings('ignore')

a = time.time()
####加载数据集
diabetes = datasets.load_diabetes()
x = pd.DataFrame()

####仅仅使用一个特征：
diabetes_X = diabetes.data[:, np.newaxis, 2]

x = pd.DataFrame(diabetes_X)
x['1'] = diabetes.target
print(x.shape)
diabetes_X_train = x[:-20].values
diabetes_X_test = x[-20:].values
print(type(diabetes_X_train))
print(diabetes_X_test.shape)
diabetes_X_train_x = diabetes_X_train[:, :-1]
diabetes_X_train_y = diabetes_X_train[:, -1]
diabetes_X_test_x = diabetes_X_test[:, :-1]
diabetes_X_test_y = diabetes_X_test[:, -1]
print(diabetes_X_train_x.shape)
print(diabetes_X_train_y.shape)
model = linear_model.LogisticRegression()




# l = [random.randint(0, 422) for i in range(10)]
#
# for i in l:
#     np.concatenate((), axis=0)
# print(l)
for i in range(4):
    w = int(422/4*i)
    wx = int(422/4*(i+1))
    # print(diabetes_X_train_x[i:i+1].shape)
    # model.fit(diabetes_X_train_x[w:wx], diabetes_X_train_y[w:wx])
    model.fit(diabetes_X_train_x[i:i+3], diabetes_X_train_y[i:i+3])
    # print(diabetes_X_train_x[w:wx].shape)
    print(mean_absolute_error(model.predict(diabetes_X_test_x), diabetes_X_test_y))
# model.fit(diabetes_X_train_x, diabetes_X_train_y)
# print(np.mean((model.predict(diabetes_X_test_x) - diabetes_X_test_y) ** 2))
# print(model.predict(diabetes_X_test_x) - diabetes_X_test_y)

# model1 = linear_model.LinearRegression()
# model1.fit(diabetes_X_train_x, diabetes_X_train_y)

# ss = x.columns.values.tolist()
# a = vc()
# a.save_version(model, diabetes_X_train, diabetes_X_test, ss)

# a.draw()if model.get_params():
#     print("OK")
# a.save_version(model)
# loadmodel = a.load_model(12)
# yy = loadmodel.predict(np.reshape(x,[len(x),1]))
# print(yy)
# a.show()