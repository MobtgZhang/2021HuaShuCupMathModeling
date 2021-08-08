import numpy as np
import torch
def onehot(vec,catagories,code_type='one-hot'):
    assert code_type in ['one-hot','dummy','effect']
    length = len(vec)
    if code_type=='one-hot':
        output = np.zeros(shape=(length,catagories))
        for k in range(length):
            output[k,vec[k]] = 1
        return output
    elif code_type=='dummy':
        output = np.zeros(shape=(length, catagories-1))
        for k in range(length):
            if vec[k] != 0:
                output[k,vec[k]-1] = 1
        return output
    elif code_type=='effect':
        output = np.zeros(shape=(length, catagories - 1))
        for k in range(length):
            if vec[k] != 0:
                output[k, vec[k] - 1] = 1
            else:
                output[k] = -1
        return output
    else:
        raise TypeError("Unknown Type %s"%str(code_type))
def vectorize(data):
    '''
    :param data: 包含有整数的一个列表
    标签b离散型数据集
    b_d_list = ['B1', 'B3', 'B6', 'B9', 'B11', 'B12']
    B1 含有3个属性
    B3 含有6个属性
    B9 含有7个属性
    B11 含有9个属性
    B12 含有11个属性
    :return:
    '''
    _,m_num = data.shape
    for k in range(m_num):
        output = onehot(data[:,k])


