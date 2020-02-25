import queue
import pandas as pd
import numpy as np
from scipy import stats


def entropy_scipy(data, att_name):
    n = data.shape[0]
    values = data[att_name].value_counts()
    return stats.entropy(values/n)


def entropy(data: pd.DataFrame, att_name):
    levels = data[att_name].unique()
    # 信息熵
    ent = 0
    for lv in levels:
        pi = sum(data[att_name] == lv) / data.shape[0]
        ent += pi*np.log(pi)
    return -ent


def conditional_entropy(data, xname, yname):
    xs = data[xname].unique()
    ys = data[yname].unique()
    p_x = data[xname].value_counts() / data.shape[0]
    ce = 0
    for x in xs:
        ce += p_x[x]*entropy_scipy(data[data[xname] == x], yname)
    return ce


def gain(data, xname, yname):
    en = entropy_scipy(data, yname)
    ce = conditional_entropy(data, xname, yname)
    return en - ce


def gain_ratio(data, xname, yname):
    g = gain(data, xname, yname)
    si = entropy_scipy(data, xname)
    return g / si


class data_tree:
    def __init__(self, data: pd.DataFrame, label):
        self.data = data
        self.label = label
        columns = self.data.columns.values.tolist()
        self.q = queue.Queue()
        for iter in columns:
            self.q.put(iter)

    def print_all(self):
        print(self.data)
        print('---------------------')
        while not self.q.empty():
            print(self.q.get())

    def search(self):
        i = 0
        while not self.q.empty():
            i = i+1
            # print('---------')
            node = self.q.get()
            if not self.data[node].isna().any():
                print('rand '+str(i)+':'+node+'---->'+str(gain_ratio(self.data, node, self.label)))
            else:
                print('rand '+str(i)+':'+node+'----> is nan')
