import pandas as pd
import itertools


class finder:
    def __init__(self, data: pd.DataFrame, group_list: list, combine_list: list):
        self.data = data
        self.group_list = group_list
        self.combine_list = combine_list
        self.iter = 0

    def f(self, row, group_name, m, n):
        return self.data[(self.data[group_name] == row[group_name]) & (row[m] < self.data[n])]

    def creat(self):
        clist = itertools.product(self.combine_list, self.combine_list)
        for group_name in self.group_list:
            for m, n in clist:
                self.data[m+n] = self.data.apply(lambda row: self.f(row, group_name, m, n), axis=1)

