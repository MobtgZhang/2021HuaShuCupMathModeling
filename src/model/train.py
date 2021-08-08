import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from .data import StatisticsDataset,batchify
from .IGNN import InteractiveGateDNNMlp

class trainer:
    def __init__(self,args):
        self.processed_path = args.processed_path
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.vec_type = args.vec_type
        self.train_times = args.train_times
        self.test_file = args.test_file
        self.use_gpu = args.cuda and torch.cuda.is_available()
    def run(self):
        filename = os.path.join(self.processed_path, 'train', "dataset.npz")
        results_path = os.path.join(self.processed_path, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        # 训练数据集和测试数据集
        vNpFile = np.load(filename, allow_pickle=True)
        train_data = vNpFile['train_data']
        test_data = vNpFile['test_data']
        train_dataset = StatisticsDataset(train_data)
        # 准备好对应的dataloader
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  shuffle=True, collate_fn=batchify)
        # 模型准备
        device = 'cuda' if self.use_gpu else 'cpu'
        model = InteractiveGateDNNMlp()
        model.to(device)

        lossfn = nn.BCELoss()
        optimizer = optim.Adamax(model.parameters())

        test_dataset = StatisticsDataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size,
                                     shuffle=False, collate_fn=batchify)
        test_loss_list = []
        test_peason_list = []
        train_loss_list = []
        train_peason_list = []
        best_test_ps = 0
        for step in range(self.train_times):
            train_loss = 0
            for (a_data, b_c_data, b_d_data), pred_data in train_loader:
                a_data, b_c_data, b_d_data = torch.tensor(a_data, dtype=torch.float), \
                                             torch.tensor(b_c_data, dtype=torch.float), \
                                             torch.tensor(b_d_data, dtype=torch.long)
                target = torch.tensor(pred_data, dtype=torch.float)
                a_data = a_data.to(device)
                b_c_data = b_c_data.to(device)
                b_d_data = b_d_data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                model.train()
                predict = model(a_data, b_c_data, b_d_data)

                loss = lossfn(predict, target)
                train_loss += loss.cpu().data.item()

                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            # 测试集测试过程
            (test_ps, test_loss), (train_target, train_predict) = \
                evaluate(model, test_dataloader, lossfn, device)

            (train_ps, train_loss), (test_target, test_predict) = \
                evaluate(model, train_loader, lossfn, device)
            if step%100 == 0:
                print("step:%d;test pearson_score:%0.4f,test_loss:%0.4f;train pearson_score:%0.4f,test_loss:%0.4f"
                  % (step,test_ps, test_loss, train_ps, train_loss))
            if best_test_ps<test_ps and test_ps>0.55:
                best_test_ps = test_ps
                # 保存模型文件
                model_file = os.path.join(results_path, "model.ckpt")
                model.to('cpu')
                torch.save(model, model_file)
                model.to(device)

            test_loss_list.append(test_loss)
            test_peason_list.append(test_ps)
            train_loss_list.append(train_loss)
            train_peason_list.append(train_ps)
        # 保存最好的数值
        best_test_file = os.path.join(results_path,'best_test.txt')
        with open(best_test_file,mode='w',encoding='utf-8') as wfp:
            wfp.write("The best pearson score of test dataset is %f"%best_test_ps)
        # 画图
        x = np.array(range(len(test_loss_list)))
        plt.plot(x,test_loss_list,c='red',label='test')
        plt.plot(x, train_loss_list, c='blue',label='train')
        plt.xlabel("训练次数")
        plt.ylabel("损失值大小")
        plt.legend()
        plt.title("测试数据集损失值变化图")

        fig_file = os.path.join(results_path,'loss.png')
        plt.savefig(fig_file)
        plt.show()
        plt.close()

        plt.plot(x, test_peason_list,  c='red',label ='test')
        plt.plot(x, train_peason_list, c='blue', label='train')
        plt.xlabel("训练次数")
        plt.ylabel("Pearson指数")
        plt.title("Pearson指数值变化图")
        plt.legend()
        fig_file = os.path.join(results_path, 'pearson.png')
        plt.savefig(fig_file)
        plt.show()
        plt.close()
    def predict(self):
        '''
        test_data = {"X": [a_data, b_c_data, b_d_data],
                     "length": len(a_data),
                     'index': user_indexes}
        :return:
        '''
        split_path = os.path.join(self.processed_path, 'train')
        filename = os.path.join(split_path, "test_dataset.npz")
        test_data = np.load(filename,allow_pickle=True)
        test_data = test_data['test_data']
        dataset = test_data.tolist()
        length = dataset["length"]
        a_data = dataset["X"][0]
        b_c_data = dataset["X"][1]
        b_d_data = dataset["X"][2]
        user_indexes = dataset['index']
        a_data, b_c_data, b_d_data = torch.tensor(a_data, dtype=torch.float), \
                                     torch.tensor(b_c_data, dtype=torch.float), \
                                     torch.tensor(b_d_data, dtype=torch.long)
        device = 'cuda' if self.use_gpu else 'cpu'
        a_data = a_data.to(device)
        b_c_data = b_c_data.to(device)
        b_d_data = b_d_data.to(device)
        # 预测数值
        results_path = os.path.join(self.processed_path, "results")
        model_file = os.path.join(results_path, "model.ckpt")
        model = torch.load(model_file)
        model.to(device)
        pred_data = model(a_data, b_c_data, b_d_data)
        pred_data = pred_data.cpu().detach().numpy().squeeze()
        all_data = pd.read_excel(self.test_file)
        for index,value in zip(range(length),pred_data):
            if np.isnan(value):
                all_data.loc[index, '是否会购买？'] = 100.00
            else:
                all_data.loc[index, '是否会购买？'] = round(value*100,2)
        results_path = os.path.join(self.processed_path, "results")
        save_file = os.path.join(results_path, "predict_results.xlsx")
        all_data.to_excel(save_file,index=False)
# 测试的指标使用皮尔森指数
def peason(target,predict):
    target = target - target.mean()
    predict = predict - predict.mean()
    return np.dot(target,predict)/(np.linalg.norm(target,ord=2)*np.linalg.norm(predict,ord=2))
def evaluate(model,data_loader,lossfn,device):
    model.eval()
    test_loss = 0
    predict = []
    target = []
    for (a_data, b_c_data, b_d_data), pred_data in data_loader:
        a_data, b_c_data, b_d_data = torch.tensor(a_data, dtype=torch.float), \
                                     torch.tensor(b_c_data, dtype=torch.float), \
                                     torch.tensor(b_d_data, dtype=torch.long)
        target_tp = torch.tensor(pred_data, dtype=torch.float)
        a_data = a_data.to(device)
        b_c_data = b_c_data.to(device)
        b_d_data = b_d_data.to(device)
        target_tp = target_tp.to(device)
        predict_tp = model(a_data, b_c_data, b_d_data)

        loss = lossfn(predict_tp, target_tp)
        test_loss += loss.cpu().data.item()
        predict.append(predict_tp.cpu().detach().numpy())
        target.append(target_tp.cpu().detach().numpy())
    predict = np.vstack(predict).squeeze()
    target = np.vstack(target).squeeze()
    peason_score = peason(target,predict)
    return (peason_score,test_loss),(target,predict)