from GlobalVar import global_nums as gln
import pandas as pd
import numpy as np
import os
import joblib
import shutil
from matplotlib import pyplot as plt



def load_file(num):
    # 创建版本对应文件夹
    if not os.path.exists('.\\file'):
        os.makedirs('.\\file')
    os.makedirs('.\\file\\%s_version' % num)


def save_model(model, n):
    # 当前版本模型保存
    path = './file/' + str(n) + '_version/Model.model'
    joblib.dump(model, path)


def load_model(num):
    # 加载版本号为num的模型
    path = './file/%s_version/' % num + 'Model.model'
    return joblib.load(path)


def load_traindata(n, data, name):
    # 保存训练集
    path = './file/' + str(n) + '_version/train.csv'
    temp = pd.DataFrame(data, columns=name)
    temp.to_csv(path)
    return path


def load_validationdata(n, data, name):
    # 保存验证集
    path = './file/' + str(n) + '_version/validation.csv'
    temp = pd.DataFrame(data, columns=name)
    temp.to_csv(path)
    return path


def load_testdata(n, data, name):
    # 保存测试集
    path = './file/' + str(n) + '_version/test.csv'
    temp = pd.DataFrame(data, columns=name)
    temp.to_csv(path)
    return path


def strtolist(data):
    #
    str = data[1:-1]
    l = str.spilt(',')
    for i in len(l):
        l[i] = float(l[i])
    return l


def model_param(model):
    # 返回一个参数信息的字典
    return model.get_params()


def model_name(model):
    # 返回模型所用方法的字符串
    return model.__class__.__name__


def result(model, test, validation=None):
    # 传入的数据集（dataframe）为完整的，即变量在前面，结果在最后一列
    return [np.mean((model.predict(test[:, :-1]) - test[:, -1:]) ** 2),
            model.score(test[:, :-1], test[:, -1:])]



class version_contrl:
    def __init__(self):
        self.a = gln()

    def save_version(self, model, train, test, feature_name, name=None, validation=None):
        n = self.a.vno
        try:
            try:
                model_param(model)
                load_file(n)
                l = result(model, test)
                if validation is None:
                    lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name), None,
                             model_name(model), model_param(model), l[0], l[1]]
                else:
                    lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name),
                             load_validationdata(n, validation, feature_name), model_name(model), model_param(model),
                             l[0], l[1]]
                data = pd.DataFrame([lists],
                                    columns=['Vno', 'train', 'test', 'Validation', 'Function', 'Param', 'mse', 'R2'])
                save_model(model, n)
                self.a.write_in_file(data)
                self.a.vno = self.a.vno + 1
            except:
                if name is None:
                    print('please Name your NN model')
                else:
                    load_file(n)
                    l = result(model, test)
                    if validation is None:
                        lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name), None,
                                 name, model_param(model), l[0], l[1]]
                    else:
                        lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name),
                                 load_validationdata(n, validation, feature_name), name,
                                 model_param(model),
                                 l[0], l[1]]
                    data = pd.DataFrame([lists],
                                        columns=['Vno', 'train', 'test', 'Validation', 'Function', 'Param', 'mse',
                                                 'R2'])
                    save_model(model, n)
                    self.a.write_in_file(data)
                    self.a.vno = self.a.vno + 1
        except:
            if os.path.exists('.\\file\\%s_version' % n):
                shutil.rmtree('.\\file\\%s_version' % n)
            print('创建版本失败')

    def show(self):
        # 打印所有版本各个信息
        print(self.a.data_now)

    def drawall(self):
        a = self.a.data_now
        plt.subplot(211)
        l = []
        values = []
        for i , v in a['mse'].iteritems():
            l.append(i)
            values.append(v)
        plt.plot(l,values)
        plt.scatter(l, values)
        plt.title('mse results')
        plt.xlabel('vno')
        plt.ylabel('mse')
        plt.subplot(212)
        l = []
        values = []
        for i, v in a['R2'].iteritems():
            l.append(i)
            values.append(v)
        plt.plot(l, values)
        plt.scatter(l, values)
        plt.title('R2 results')
        plt.xlabel('vno')
        plt.ylabel('R2')
        plt.show()


