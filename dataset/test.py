import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

root_dir = os.getcwd()
f_lists = os.listdir(root_dir)
print("File Lists : ", f_lists)

new_file_lists = [f for f in f_lists if f.endswith('.csv')]
print("File Lists : ", new_file_lists)

data_lists = new_file_lists[:-1]
error_list = new_file_lists[-1]
print("Data Lists : ", data_lists)
print("Error Data List : ", error_list)

def csv_read_(data_dir, data_list):
    tmp = pd.read_csv(os.path.join(data_dir, data_list), sep=',', encoding='UTF8')
    y, m, d =map(int, data_list.split('-')[-1].split('.')[:-1])
    time = tmp['Time']
    tmp['DTime'] ='-'.join(data_list.split('-')[-1].split('.')[:-1])
    ctime = time.apply(lambda _ : _.replace(u'오후', 'PM').replace(u'오전', 'AM'))
    n_time = ctime.apply(lambda _ : datetime.datetime.strptime(_, "%p %I:%M:%S"))
    newtime = n_time.apply(lambda _ : _.replace(year=y, month=m, day=d))
    tmp['Time'] = newtime
    return tmp

dd = csv_read_(root_dir, data_lists[0])
for i in range(1, len(data_lists)):
    dd = pd.merge(dd, csv_read_(root_dir, data_lists[i]), how='outer')

dd = dd.drop('Index', axis=1)

dd = dd.set_index('Time')

dedicated_data = dd.copy()

dedicated_data = dedicated_data.dropna()

lot_lists = dedicated_data['LoT'].unique()

d_lists = dedicated_data['DTime'].unique()

process = pd.read_csv(os.path.join(root_dir, error_list), sep=',', encoding='utf-8')
lot_process_lists = process['LoT'].unique()
d_process_lists = process['Date'].unique()
X_data = pd.DataFrame(columns={'pH','Temp', 'LoT', 'Process'})
for d in d_lists:
    for lot in lot_lists:
        tmp = dedicated_data[(dedicated_data['DTime']==d)&(dedicated_data['LoT']==lot)]
        tmp = tmp[['pH', 'Temp', 'LoT']]
        process_val = process[(process['Date']==d)&((process['LoT']==lot))]['Process Rate'].values
        trr = np.full((tmp['pH'].shape), process_val)
        tmp['Process'] = trr
        X_data = X_data.append(tmp)
X_data=X_data.apply(pd.to_numeric)
X_data = X_data[['LoT', 'pH', 'Temp', 'Process']]

X_data.to_csv("testdata.csv")