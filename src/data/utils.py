import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
def filter_file(filename,filter_path):
    all_data = pd.read_excel(filename)
    a_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
    indexes = False
    for item in a_list:
        indexes = ((all_data.loc[:, item] > 100) | (all_data.loc[:, item] < 0) ) | indexes
    error_a_data = all_data[indexes]
    error_a_file = os.path.join(filter_path,'errors_a.xlsx')
    error_a_data.to_excel(error_a_file,index=False)

    error_b_data = all_data[pd.isnull(all_data.loc[:,'B7'])]
    error_b_file = os.path.join(filter_path, 'errors_b.xlsx')
    error_b_data.to_excel(error_b_file, index=False)

    # 修正数值B7
    length = len(all_data)
    for k in range(length):
        if pd.isnull(all_data.loc[k,'B7']):
            all_data.loc[k, 'B7'] = 0
    # 修正数值a
    for k in range(length):
        for a_tag in a_list:
            if all_data.loc[k,a_tag]>100:
                all_data.loc[k,a_tag] = all_data.loc[k,a_tag]/10
    filter_file1 = os.path.join(filter_path,'filter_data1.xlsx')
    all_data.to_excel(filter_file1, index=False)
def filter_split_file(filename,filter_path):
    all_data = pd.read_excel(filename)
    for k in range(1,4):
        type_id_data = all_data[all_data['品牌类型'] == k]
        type_id_file = os.path.join(filter_path,'type_%d.xlsx'%k)
        type_id_data.to_excel(type_id_file, index=False)

def create_dataset(filename,processed_path,percentage):
    # 准备数据集
    all_data = pd.read_excel(filename)
    # 标签a数据集准备
    a_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
    # 标签b连续型数据集准备
    b_z_score = ['B2','B4','B8','B10','B13','B14','B15']
    b_max_min = ['B5','B7']
    b_other = ['B16','B17']
    b_c_list = ['B2', 'B4', 'B5', 'B7', 'B8', 'B10', 'B13', 'B14', 'B15', 'B16', 'B17']
    # 标签b离散型数据集准备
    b_d_list = ['B1', 'B3', 'B6', 'B9', 'B11', 'B12']
    split_path = os.path.join(processed_path, 'train')
    for num in range(1,4):
        data = all_data[all_data['品牌类型'] == num]
        data_length = len(data)
        pred_data = data.loc[:,'购买意愿']
        a_data = data.loc[:, a_list]
        a_data = a_data / 100.0
        b_c_data = data.loc[:,b_c_list]

        b_c_data.loc[:, b_max_min] = (b_c_data.loc[:, b_max_min]-b_c_data.loc[:, b_max_min].min())\
                                    /(b_c_data.loc[:, b_max_min].max()-b_c_data.loc[:, b_max_min].min())

        b_c_data.loc[:,b_z_score] = (b_c_data.loc[:,b_z_score]-b_c_data.loc[:,b_z_score].mean())\
                                    /b_c_data.loc[:,b_z_score].std()
        b_c_data.loc[:,b_other] /= 100

        b_d_data = data.loc[:, b_d_list]

        for index,item in b_d_data.iterrows():
            value = b_d_data.loc[index, 'B9']
            if value == 8:
                b_d_data.loc[index, 'B9'] = 7
        b_d_data = b_d_data - 1
        # 将数据分为训练数据和测试数据
        train_len = int(percentage*data_length)
        # 将数据打乱
        pred_data = pred_data.values.reshape(data_length,1)
        values = [a_data.values,b_c_data.values,b_d_data.values,pred_data]
        output = np.hstack(values)
        output = shuffle(output)
        # 分开训练数据和测试数据
        a_len = len(a_list)
        b_c_len = len(b_c_list)
        b_d_len = len(b_d_list)
        a_data = output[:,:a_len]
        b_c_data = output[:,a_len:a_len+b_c_len]
        b_d_data = output[:,a_len+b_c_len:a_len+b_c_len+b_d_len]
        pred_data = output[:,-1].reshape(data_length,1)
        train_a_data = a_data[:train_len]
        test_a_data = a_data[train_len:]
        train_b_c_data = b_c_data[:train_len]
        test_b_c_data = b_c_data[train_len:]
        train_b_d_data = b_d_data[:train_len]
        test_b_d_data = b_d_data[train_len:]
        train_pred_data = pred_data[:train_len]
        test_pred_data = pred_data[train_len:]
        train_data = {"X":[train_a_data,train_b_c_data,train_b_d_data],
                      "Y":train_pred_data,
                      "length":len(train_pred_data)}
        test_data = {"X":[test_a_data,test_b_c_data,test_b_d_data],
                     "Y":test_pred_data,
                     "length":len(test_pred_data)}
        print("train dataset length:",len(train_pred_data))
        print("test dataset length:",len(test_pred_data))
        filename = os.path.join(split_path,"dataset_type%d.npz"%num)
        np.savez(filename,train_data=train_data,test_data=test_data)
def create_all_dataset(filename,processed_path,percentage):
    # 准备数据集
    all_data = pd.read_excel(filename)
    # 标签a数据集准备
    a_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
    # 标签b连续型数据集准备
    b_z_score = ['B2', 'B4', 'B8', 'B10', 'B13', 'B14', 'B15']
    b_max_min = ['B5', 'B7']
    b_other = ['B16', 'B17']
    b_c_list = ['B2', 'B4', 'B5', 'B7', 'B8', 'B10', 'B13', 'B14', 'B15', 'B16', 'B17']
    # 标签b离散型数据集准备
    b_d_list = ['B1', 'B3', 'B6', 'B9', 'B11', 'B12','品牌类型']

    split_path = os.path.join(processed_path, 'train')
    data_length = len(all_data)
    pred_data = all_data.loc[:, '购买意愿']
    user_indexes = all_data.loc[:,'目标客户编号']
    a_data = all_data.loc[:, a_list]
    a_data = a_data / 100.0
    b_c_data = all_data.loc[:, b_c_list]

    b_c_data.loc[:, b_max_min] = (b_c_data.loc[:, b_max_min] - b_c_data.loc[:, b_max_min].min()) \
                                 / (b_c_data.loc[:, b_max_min].max() - b_c_data.loc[:, b_max_min].min())

    b_c_data.loc[:, b_z_score] = (b_c_data.loc[:, b_z_score] - b_c_data.loc[:, b_z_score].mean()) \
                                 / b_c_data.loc[:, b_z_score].std()
    b_c_data.loc[:, b_other] /= 100

    b_d_data = all_data.loc[:, b_d_list]

    for index, item in b_d_data.iterrows():
        value = b_d_data.loc[index, 'B9']
        if value == 8:
            b_d_data.loc[index, 'B9'] = 7
    b_d_data = b_d_data - 1
    # 将数据分为训练数据和测试数据
    train_len = int(percentage * data_length)
    # 将数据打乱
    pred_data = pred_data.values.reshape(data_length, 1)
    user_indexes = user_indexes.values.reshape(data_length, 1)
    values = [a_data.values, b_c_data.values, b_d_data.values, pred_data,user_indexes]
    output = np.hstack(values)
    output = shuffle(output)
    # 分开训练数据和测试数据
    a_len = len(a_list)
    b_c_len = len(b_c_list)
    b_d_len = len(b_d_list)
    a_data = output[:, :a_len]
    b_c_data = output[:, a_len:a_len + b_c_len]
    b_d_data = output[:, a_len + b_c_len:a_len + b_c_len + b_d_len]
    pred_data = output[:, -2].reshape(data_length, 1)
    user_indexes = output[:, -1].reshape(data_length, 1)
    train_a_data = a_data[:train_len]
    test_a_data = a_data[train_len:]
    train_b_c_data = b_c_data[:train_len]
    test_b_c_data = b_c_data[train_len:]
    train_b_d_data = b_d_data[:train_len]
    test_b_d_data = b_d_data[train_len:]
    train_pred_data = pred_data[:train_len]
    train_indexes = user_indexes[:train_len]
    test_pred_data = pred_data[train_len:]
    test_indexes = user_indexes[train_len:]
    train_data = {"X": [train_a_data, train_b_c_data, train_b_d_data],
                  "Y": train_pred_data,
                  "length": len(train_pred_data),
                  'index':train_indexes}
    test_data = {"X": [test_a_data, test_b_c_data, test_b_d_data],
                 "Y": test_pred_data,
                 "length": len(test_pred_data),
                 'index':test_indexes}
    print("train dataset length:", len(train_pred_data))
    print("test dataset length:", len(test_pred_data))
    filename = os.path.join(split_path, "dataset.npz")
    np.savez(filename, train_data=train_data, test_data=test_data)

def create_test_dataset(filename,processed_path):
    # 准备数据集
    all_data = pd.read_excel(filename)
    # 标签a数据集准备
    a_list = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']
    # 标签b连续型数据集准备
    b_z_score = ['B2', 'B4', 'B8', 'B10', 'B13', 'B14', 'B15']
    b_max_min = ['B5', 'B7']
    b_other = ['B16', 'B17']
    b_c_list = ['B2', 'B4', 'B5', 'B7', 'B8', 'B10', 'B13', 'B14', 'B15', 'B16', 'B17']
    # 标签b离散型数据集准备
    b_d_list = ['B1', 'B3', 'B6', 'B9', 'B11', 'B12', '品牌编号 ']


    user_indexes = all_data.loc[:, '客户编号']
    a_data = all_data.loc[:, a_list]
    a_data = a_data / 100.0
    b_c_data = all_data.loc[:, b_c_list]

    b_c_data.loc[:, b_max_min] = (b_c_data.loc[:, b_max_min] - b_c_data.loc[:, b_max_min].min()) \
                                 / (b_c_data.loc[:, b_max_min].max() - b_c_data.loc[:, b_max_min].min())

    b_c_data.loc[:, b_z_score] = (b_c_data.loc[:, b_z_score] - b_c_data.loc[:, b_z_score].mean()) \
                                 / b_c_data.loc[:, b_z_score].std()
    b_c_data.loc[:, b_other] /= 100

    b_d_data = all_data.loc[:, b_d_list]

    for index, item in b_d_data.iterrows():
        value = b_d_data.loc[index, 'B9']
        if value == 8:
            b_d_data.loc[index, 'B9'] = 7
    b_d_data = b_d_data - 1

    a_data = a_data.values
    b_c_data = b_c_data.values
    b_d_data = b_d_data.values
    user_indexes = user_indexes.values
    test_data = {"X": [a_data, b_c_data,b_d_data],
                 "length": len(a_data),
                 'index': user_indexes}
    split_path = os.path.join(processed_path, 'train')
    filename = os.path.join(split_path, "test_dataset.npz")
    np.savez(filename, test_data = test_data)
