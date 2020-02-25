from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import joblib
import os
# import reportgen as rpt
path: str = './model/'


def pd2arr(field):
    """
    pandas转为列向量
    :param field: pandas列
    :return: 列向量
    """
    return np.array([field]).T


def standard_scale(data, field_name):
    """
    标准化
    :param data: 数据集
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = pd2arr(data[field_name])
    file_path = path+field_name+"_StandardScaler.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.StandardScaler()
        scale.fit(x)
        joblib.dump(scale, file_path)
    data[field_name] = scale.transform(x)
    return data[field_name]


def min_max_scale(data, field_name):
    """
    min-max标准化
    :param data: 数据集
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = pd2arr(data[field_name])
    file_path = path + field_name + "_MinMaxScaler.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.MinMaxScaler()
        scale.fit(x)
        joblib.dump(scale, file_path)
    data[field_name] = scale.transform(x)
    return data[field_name]


def max_abs_scale(data, field_name):
    """
    最大值标准化
    :param data: 数据集
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = pd2arr(data[field_name])
    file_path = path + field_name + "_MaxAbsScaler.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.MaxAbsScaler()
        scale.fit(x)
        joblib.dump(scale, file_path)
    data[field_name] = scale.transform(x)
    return data[field_name]


def normalizer(data, field_name, norm='l2'):
    """
    规范化
    :param data: 数据集
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = np.array([data[field_name]])
    file_path = path + field_name + "_Normalizer.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.Normalizer()
        scale.fit(x)
        joblib.dump(scale, file_path)
    # data[field_name] = pd.DataFrame(scale.transform(x))
    return scale.transform(x)


def binarizer(data, field_name):
    """
    二值化
    :param data:
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = pd2arr(data[field_name])
    file_path = path + field_name + "_Binarizer.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.Binarizer()
        scale.fit(x)
        joblib.dump(scale, file_path)
    data[field_name] = scale.transform(x)
    return data[field_name]


def one_hot_encoder(data, field_name):
    """
    将类别onthot处理
    :param data:
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = pd2arr(data[field_name])
    file_path = path + field_name + "_OneHotEncoder.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.OneHotEncoder(categories='auto')
        scale.fit(x)
        joblib.dump(scale, file_path)
    tmp = scale.transform(x).toarray()
    columns = [field_name+str(i) for i in range(tmp.shape[1])]
    new_time = pd.DataFrame(tmp, columns=columns)
    data[columns] = new_time
    # print('111')
    # del data[field_name]
    # print(data.join(new_time))
    return new_time


def label_encoder(data, field_name):
    """
    将类别转为数值
    :param data:
    :param field_name: 字段名
    :return: 变化后的列
    """
    x = np.array(data[field_name])
    # print(pd2arr(data[field_name]).shape)
    file_path = path + field_name + "_LabelEncoder.save"
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.LabelEncoder()
        scale.fit(x)
        joblib.dump(scale, file_path)
    data[field_name+'_label'] = scale.transform(x)
    return data[field_name+'_label']


def pca(data, n):
    """
    PAC降维
    :param data: 全部特征
    :param n: 降成n维
    :return: 降维后的特征
    """
    x = np.array(data)
    file_path = path + "PCA_" + n + "_LabelEncoder.save"
    if os.path.exists(file_path):
        model = joblib.load(file_path)
    else:
        model = PCA(n_components=n)
        model.fit_transform(x)
        joblib.dump(model,file_path)
    return model.transform(x)


def lda(data, y, n):
    """
    LDA降维
    :param data: 全部特征
    :param y: 标签
    :param n: 降成n维
    :return: 降维后的特征
    """
    x = np.array(data)
    file_path = path + "LDA_" + n + "_LabelEncoder.save"
    if os.path.exists(file_path):
        model = joblib.load(file_path)
    else:
        model = LDA(n_components=n)
        model.fit(x, y)
        joblib.dump(model, file_path)
    return model.transform(x)


def imputer(data, field_name, strategy):
    """
    缺失值填充
    :param data: 数据集
    :param field_name: 字段名称
    :param strategy: 填充策略
    :return: 填充完成的字段
    """
    x = pd2arr(data[field_name])
    file_path = path + field_name + "_OneHotEncoder.save"
    if os.path.exists(file_path):
        na = joblib.load(file_path)
    else:
        na = SimpleImputer(missing_values=np.nan, strategy=strategy, verbose=0)
        na.fit(x)
        joblib.dump(na, file_path)
    data[field_name] = na.transform(x)
    return data[field_name]


def time_cange(data, field_name):
    """
    处理时间特征
    :param field_name:
    :param data: 数据集
    :return:
    """
    x = data[field_name]
    temp = pd.DatetimeIndex(x)
    data['year'] = temp.year
    data['month'] = temp.month
    data['day'] = temp.day
    data['hour'] = temp.hour
    data['minute'] = temp.minute
    data['second'] = temp.second
    # data.drop(columns=X.name, inplace=True)
    del data[field_name]
    return data


def discretization(data, field_name, label_name, method, max_intervals):
    x = np.array(data[field_name])
    y = np.array(data[label_name])
    file_path = path + field_name + "_Discretization.save"
    if os.path.exists(file_path):
        dis = joblib.load(file_path)
    # else:
        # dis = rpt.preprocessing.Discretization(method=method, max_intervals=max_intervals)
        # dis.fit(x, y)
        # joblib.dump(dis, file_path)
    tmp = np.array(dis.transform(x))
    tmp = [str(i) for i in tmp]
    file_path = path + field_name + "_Discretization" + "_LabelEncoder.save"
    if os.path.exists(file_path):
        encoder = joblib.load(file_path)
    else:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(tmp)
        joblib.dump(encoder,file_path)
    data[field_name] = encoder.transform(tmp)
    return data[field_name]


def robust_scale(data, field_name):
    x = pd2arr(data[field_name])
    file_path = path + field_name + '_RobustScale.save'
    if os.path.exists(file_path):
        scale = joblib.load(file_path)
    else:
        scale = preprocessing.RobustScaler()
        scale.fit(x)
        joblib.dump(scale, file_path)
    data[field_name] = scale.transform(x)
    return data[field_name]


