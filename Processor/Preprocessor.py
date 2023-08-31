#!/usr/bin/env python
# coding: utf-8

# In[17]:


from scipy import interpolate
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split


# In[18]:


def create_empty_lists(count):
    empty_lists = [list() for _ in range(count)]
    return empty_lists


# In[19]:


def generate_list(num):
    my_list = ['normal'] + [f'type{i}' for i in range(1, num)]
    return my_list


# In[20]:


def PreProcessing(Path, class_num = 4,num_sensors = 4, data_lenth = 140, sampling_period= 0.001):
    sensor_list = []
    # Path 에 csv file 모두 list에 load
    item_list = generate_list(class_num)
    for filename in os.listdir(Path):
        if filename.endswith('.csv'):
            sensor = pd.read_csv(Path+filename,names = ['time']+item_list)
            sensor_list.append(sensor)
    
    #sensor 간 sampling rate가 다른 경우 interpolation을 통해 맞춰줍니다.
    x_new = np.arange(0, data_lenth, sampling_period)    
    y_new = [[] for _ in range(num_sensors)]
    for sensor_idx in range(num_sensors):
        sensor = sensor_list[sensor_idx]
        for item in item_list:
            f_linear = interpolate.interp1d(sensor['time'], sensor[item], kind = 'linear')
            y_new[sensor_idx].append(f_linear(x_new))
    
    #class 별로 데이터 분류
    sensor_dfs = []
    for sensor_idx, sensor in enumerate(sensor_list):
        sensor_df = pd.DataFrame(np.array(y_new[sensor_idx]).T, columns = item_list)
        sensor_dfs.append(sensor_df)
        
    concatenated_dfs = []
    for item in item_list:
        df_columns = [f's{i+1}' for i in range(num_sensors)]
        concatenated_df = pd.concat([sensor_df[item] for sensor_df in sensor_dfs], axis = 1)
        concatenated_df.columns = df_columns
        concatenated_dfs.append(concatenated_df.values)
    
    
    # train : 정상 데이터 60%
    # valid : 정상 데이터 20%
    # test  : 정상 데이터 20% 이상데이터 100%
    train_loader, valid_loader, test_loader = get_data(concatenated_dfs[0], *concatenated_dfs[1:],sensor_num = num_sensors)
    
    return train_loader, valid_loader, test_loader


# In[21]:


def get_data(*data,sensor_num):
    window_size = 100
    data = [torch.FloatTensor(d).reshape(-1, window_size, sensor_num) for d in data]
    
    train_x, test_x = train_test_split(data[0], test_size=0.2, shuffle=False)
    train_x, valid_x = train_test_split(train_x, test_size=0.2, shuffle=False)
    train_y = torch.full((len(train_x),), 0)
    valid_y = torch.full((len(valid_x),), 0)
    test_y = torch.full((len(test_x),), 0)

    type_y = [torch.full((len(d),), i+1) for i, d in enumerate(data[1:])]

    train_set = TensorDataset(train_x, train_y)
    valid_set = TensorDataset(valid_x, valid_y)
    test_set = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)

    type_sets = [TensorDataset(d, y) for d, y in zip(data[1:], type_y)]
    test_set_all = ConcatDataset([test_set] + type_sets)
    test_loader = DataLoader(test_set_all, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader




