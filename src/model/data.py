import numpy as np
import torch.utils.data as data
class StatisticsDataset(data.Dataset):
    def __init__(self,dataset):
        '''
        :param dataset: 数据集
        '''
        # train_data = {"X":[train_a_data,train_b_c_data,train_b_d_data],
        #                       "Y":train_pred_data,
        #                       "length":train_len}
        self.dataset = dataset.tolist()
        self.length = self.dataset["length"]
    def __getitem__(self, item):
        a_data = self.dataset["X"][0][item]
        b_c_data = self.dataset["X"][1][item]
        b_d_data = self.dataset["X"][2][item]
        pred_data = self.dataset["Y"][item]
        return (a_data, b_c_data, b_d_data), pred_data

    def __len__(self):
        return self.length
def batchify(batch):
    a_data = []
    b_c_data = []
    b_d_data = []
    pred_data = []
    for item in batch:
        pred_data.append(item[-1])
        a_data.append(item[0][0])
        b_c_data.append(item[0][1])
        b_d_data.append(item[0][2])
    a_data = np.vstack(a_data)
    b_c_data = np.vstack(b_c_data)
    b_d_data = np.vstack(b_d_data)
    pred_data = np.vstack(pred_data)
    return (a_data,b_c_data,b_d_data),pred_data