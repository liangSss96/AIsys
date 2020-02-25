from VersionControl import version_contrl as vc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import time
import pandas as pd

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
print(diabetes_X_test.shape)
diabetes_X_train_x = diabetes_X_train[:, :-1]
diabetes_X_train_y = diabetes_X_train[:, -1:]
diabetes_X_test_x = diabetes_X_test[:, :-1]
diabetes_X_test_y = diabetes_X_test[:, -1:]
print(diabetes_X_train_x.shape)
print(diabetes_X_train_y.shape)
model = linear_model.LogisticRegression()

model.fit(diabetes_X_train_x, diabetes_X_train_y)
# model1 = linear_model.LinearRegression()
# model1.fit(diabetes_X_train_x, diabetes_X_train_y)


ss = x.columns.values.tolist()
# print(type(model.__class__.__name__))
a = vc()
a.save_version(model, diabetes_X_train, diabetes_X_test, ss)

# a.draw()if model.get_params():
#     print("OK")
# a.save_version(model)
# loadmodel = a.load_model(12)
# yy = loadmodel.predict(np.reshape(x,[len(x),1]))
# print(yy)
# a.show()