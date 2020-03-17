import pandas as pd
import os

class global_nums:

    def __init__(self):
        global vno  #版本号
        global data_now  #全部版本信息
        self.vno, self.data_now = self.get_from_file()
        print("get data successfully")

    def get_from_file(self):
        # 获得版本文件中获得最大版本号和全部版本信息
        if os.path.exists('vs.csv'):
            data = pd.read_csv('vs.csv')
            if len(data['Vno'].values) == 0:
                num = 0
            else:
                num = data.iloc[-1]['Vno'] + 1
        else:
            data = pd.DataFrame(columns=['create_time', 'Vno', 'train', 'test', 'Validation', 'Function', 'Param', 'mse'])
            data.to_csv('vs.csv', index=False)
            num = 0
        return [num, data]

    def now_no(self):
        # 打印当前版本全量信息
        print('now vno', self.vno)

    def write_in_file(self, setdata):
        # 将某一版本信息写入到版本文件中
        '''
        :param setdata:某一版本的全部信息
        :return: None
        '''
        self.data_now = pd.concat([self.data_now, setdata], ignore_index=True)
        self.data_now.to_csv('vs.csv', index=False)
