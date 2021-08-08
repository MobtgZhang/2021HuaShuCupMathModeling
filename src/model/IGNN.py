import torch
import torch.nn as nn
import torch.nn.functional as F
class InteractiveGateDNNMlp(nn.Module):
    def __init__(self,emb_list=None,linear_dim=8,hidden_dim = 6):
        super(InteractiveGateDNNMlp, self).__init__()
        if emb_list is None:
            emb_list = [2, 3, 5, 6, 8, 10,2] # 这个数值最好
        assert len(emb_list) == 7
        # Embedding 区域
        self.b1_embed_dim = emb_list[0]
        self.b3_embed_dim = emb_list[1]
        self.b6_embed_dim = emb_list[2]
        self.b9_embed_dim = emb_list[3]
        self.b11_embed_dim = emb_list[4]
        self.b12_embed_dim = emb_list[5]
        self.type_embed_dim = emb_list[6]
        self.b1_embed = nn.Embedding(3,self.b1_embed_dim)
        self.b3_embed = nn.Embedding(6,self.b3_embed_dim)
        self.b6_embed = nn.Embedding(8,self.b6_embed_dim)
        self.b9_embed = nn.Embedding(7,self.b9_embed_dim)
        self.b11_embed = nn.Embedding(9, self.b11_embed_dim)
        self.b12_embed = nn.Embedding(11, self.b12_embed_dim)
        self.type_embed = nn.Embedding(3, self.type_embed_dim)
        # 线性变换
        self.linear_dim = linear_dim
        self.b1_linear = nn.Linear(self.b1_embed_dim,self.linear_dim)
        self.b3_linear = nn.Linear(self.b3_embed_dim,self.linear_dim)
        self.b6_linear = nn.Linear(self.b6_embed_dim,self.linear_dim)
        self.b9_linear = nn.Linear(self.b9_embed_dim,self.linear_dim)
        self.b11_linear = nn.Linear(self.b11_embed_dim,self.linear_dim)
        self.b12_linear = nn.Linear(self.b12_embed_dim,self.linear_dim)
        self.type_linear = nn.Linear(self.type_embed_dim,self.linear_dim)
        # 交互式计算
        self.hidden_dim = hidden_dim
        self.bais_wrt = nn.Linear(8, hidden_dim)
        self.bais_wzt = nn.Linear(11, hidden_dim)
        self.bais_wpt = nn.Linear(linear_dim*7,hidden_dim)
        self.wrt = nn.Linear(8, hidden_dim, bias=False)
        self.wzt = nn.Linear(11, hidden_dim, bias=False)
        self.wht = nn.Linear(hidden_dim*2, hidden_dim, bias=False)
        self.wpt = nn.Linear(linear_dim*7, hidden_dim, bias=False)
        # 输出层
        self.out = nn.Linear(3*hidden_dim,1)
    def forward(self,a_data,b_c_data,b_d_input):
        '''
        :param a_data: size of (batch_size,8)
        :param b_c_data: size of (batch_size,11)
        :param b_d_input: size of (batch,6)
        :return:
        '''

        o1 = torch.relu(self.b1_linear(self.b1_embed(b_d_input[:,0])))
        o3 = torch.relu(self.b3_linear(self.b3_embed(b_d_input[:,1])))
        o6 = torch.relu(self.b6_linear(self.b6_embed(b_d_input[:,2])))
        o9 = torch.relu(self.b9_linear(self.b9_embed(b_d_input[:,3])))
        o11 = torch.relu(self.b11_linear(self.b11_embed(b_d_input[:,4])))
        o12 = torch.relu(self.b12_linear(self.b12_embed(b_d_input[:,5])))
        otype = torch.relu(self.type_linear(self.type_embed(b_d_input[:,6])))
        b_d_data = torch.cat([otype,o1, o3, o6, o9, o11, o12], dim=1)  # (batch,7*linear_dim)
        # 属性对齐
        pbdt = F.relu(self.bais_wpt(b_d_data))
        pat = F.relu(self.bais_wrt(a_data))
        pbct = F.relu(self.bais_wzt(b_c_data))
        # 交互式计算
        a_tp = F.relu(self.wrt(a_data))
        bc_tp = F.relu(self.wzt(b_c_data))
        bd_tp = F.relu(self.wpt(b_d_data))
        zt = torch.tanh(self.wht(torch.cat([a_tp,bc_tp],dim=1)))
        bdt = (1-zt)*pbdt+zt*pbdt
        zt = torch.tanh(self.wht(torch.cat([a_tp,bd_tp], dim=1)))
        bct = (1-zt)*pbct+zt*pbct
        zt = torch.tanh(self.wht(torch.cat([bc_tp,bd_tp], dim=1)))
        at = (1 - zt) * pat + zt * pat
        # output layer
        cat_tensor = torch.cat([at,bct,bdt],dim=1)
        return torch.sigmoid(self.out(cat_tensor))
