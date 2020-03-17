from GlobalVar import global_nums as gln
import pandas as pd
import numpy as np
import os
import joblib
import shutil
from matplotlib import pyplot as plt
import time



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


def NN_model_param(n, model):
    # 打印神经网络模型结构、参数至文件
    f = './file/' + str(n) + '_version/param.txt'
    with open(f, 'w') as file:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        for i in model.layers:
            file.write(str(i.get_config()) + '\n')
    return f


def result(model, test, validation=None, ignore = False):
    if ~ignore:
        print(test[:, -1:].shape)
        # 传入的数据集（dataframe）为完整的，即变量在前面，结果在最后一列
        return np.mean((model.predict(test[:, :-1]) - test[:, -1:]) ** 2)
    else:
        return 0


class version_contrl:
    def __init__(self):
        self.a = gln()

    def save_version(self, model, train, test, feature_name, validation=None):
        n = self.a.vno
        try:
            tempname = model_name(model)
            name = 'Neural Network' if tempname is 'Sequential' else tempname
            load_file(n)
            if validation is None:
                if name is 'Neural Network':
                    lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name), 'None',
                            name, NN_model_param(n, model), result(model, test)]
                else:
                    lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name), 'None',
                             name, model_param(model), result(model, test)]
            else:
                if name is 'Neural Network':
                    lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name),
                            load_validationdata(n, validation, feature_name), name, NN_model_param(n, model),
                            result(model, test)]
                else:
                    lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name),
                             load_validationdata(n, validation, feature_name), name, model_param(model),
                             result(model, test)]
            lists.insert(0,time.strftime('%Y-%m-%d %H:%M:%S'))
            data = pd.DataFrame([lists],
                                columns=['create_time', 'Vno', 'train', 'test', 'Validation', 'Function', 'Param', 'mse'])
            save_model(model, n)
            self.a.write_in_file(data)
            self.a.vno = self.a.vno + 1
        except:
            if os.path.exists('.\\file\\%s_version' % n):
                shutil.rmtree('.\\file\\%s_version' % n)
            print('创建版本失败')
        # tempname = model_name(model)
        # name = 'Neural Network' if tempname is 'Sequential' else tempname
        # load_file(n)
        # if validation is None:
        #     if name is 'Neural Network':
        #         lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name), 'None',
        #                  name, NN_model_param(n, model), result(model, test)]
        #     else:
        #         lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name), 'None',
        #                  name, model_param(model), result(model, test)]
        # else:
        #     if name is 'Neural Network':
        #         lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name),
        #                  load_validationdata(n, validation, feature_name), name, NN_model_param(n, model),
        #                  result(model, test)]
        #     else:
        #         lists = [n, load_traindata(n, train, feature_name), load_testdata(n, test, feature_name),
        #                  load_validationdata(n, validation, feature_name), name, model_param(model),
        #                  result(model, test)]
        # data = pd.DataFrame([lists],
        #                     columns=['Vno', 'train', 'test', 'Validation', 'Function', 'Param', 'mse'])
        # save_model(model, n)
        # self.a.write_in_file(data)
        # self.a.vno = self.a.vno + 1

    def show(self):
        # 打印所有版本各个信息
        print(self.a.data_now)

    def drawall(self):
        a = self.a.data_now
        # plt.subplot(211)
        # l = []
        # values = []
        # for i , v in a['mse'].iteritems():
        #     l.append(i)
        #     values.append(v)
        # plt.plot(l,values)
        # plt.scatter(l, values)
        # plt.title('mse results')
        # plt.xlabel('vno')
        # plt.ylabel('mse')
        # plt.subplot(212)
        # l = []
        # values = []
        # for i, v in a['R2'].iteritems():
        #     l.append(i)
        #     values.append(v)
        # plt.plot(l, values)
        # plt.scatter(l, values)
        # plt.title('R2 results')
        # plt.xlabel('vno')
        # plt.ylabel('R2')
        # plt.show()
        l = []
        values = []
        lables = []
        for i , v in a['mse'].iteritems():
            l.append(i)
            values.append(v)
        for i, v in a['Function'].iteritems():
            lables.append(v)
        fig, ax = plt.subplots()
        plt.plot(l,values)
        ax.scatter(l, values)

        plt.title('mse results')
        plt.xlabel('vno')
        plt.ylabel('mse')
        for i, txt in enumerate(lables):
            ax.annotate(txt,(l[i],values[i]))
        plt.show()


