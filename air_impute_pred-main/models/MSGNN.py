from calendar import c
import torch
from torch import nn
from models.gru import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add #, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter
from torch_geometric.utils import dense_to_sparse, to_dense_adj


class MSGRU(nn.Module):

    def __init__(self, hist_len, pred_len, k_hop, batch_size, num_node,
                 input_dim, hid_dim, interval_set, 
                 graph_gnn, 
                 fc_out,
                 device,
                 use_met, use_time):
        super(MSGRU, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.k_hop = k_hop
        self.batch_size = batch_size 
        self.num_node = num_node
        self.hid_dim = hid_dim

        self.use_met = use_met
        self.use_time = use_time

        self.graph_gnn = graph_gnn
        self.fc_out = fc_out

        self.device = device
        
        self.gru_cell_set = nn.ModuleList([GRUCell(input_dim, hid_dim) for _ in range(k_hop)])
 
    def forward(self, model_input):
        '''
        1. 
            hist_len中x直接从x_hist获取
            在pred_len中, 需要k_hop结合预测得下一时刻的x值    
        '''
        enc, dec = model_input
        x_hist = enc[0] # [batch_size, hist_len, N, 1]
        enc_misc = enc[1:]

        if self.use_met or self.use_time:
            dec = torch.cat(dec, dim=-1)
            enc_misc = torch.cat(enc_misc, dim=-1)

        features = torch.cat([enc_misc, dec], dim=1) # [batch_size, hist_len+pred_len, N, dim_features]

        y_pred = []
        
        h0_set = []
        for k in range(self.k_hop): # 初始化hidden state
            h0_set.append(torch.zeros(self.batch_size * self.num_node, self.hid_dim).to(self.device))

        hn_set = h0_set

        for t in range(self.hist_len-1):
            xn = torch.cat([x_hist[:, t], features[:, t]], dim=-1)
            xn_gnn = xn
            xn_gnn = xn_gnn.contiguous()

            for k in range(self.k_hop):
                xn_gnn = self.graph_gnn(xn_gnn)
                x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1) # clone() -> Gradient Accumulate
                hn_set[k] = self.gru_cell_set[k](x_gru, hn_set[k])
            
        x = x_hist[:, t+1]

        for i in range(self.pred_len):
            xn = torch.cat([x, features[:, self.hist_len - 1 + i]], dim=-1)

            xn_gnn = xn
            xn_gnn = xn_gnn.contiguous()

            for k in range(self.k_hop):
                xn_gnn = self.graph_gnn(xn_gnn)
                x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
                hn_set[k] = self.gru_cell_set[k](x_gru, hn_set[k])
            
            hn_all_set = []
            for k in range(self.k_hop):
                hn_all_set.append(hn_set[k].view(self.batch_size, self.num_node, self.hid_dim))
            
            hn_all = torch.cat(hn_all_set, dim=-1) # 合并不同hop gcn后的结果
            x = self.fc_out(hn_all)
            y_pred.append(x)

        return y_pred

class GraphGNN(nn.Module): # update edges by 2-FC, then edges update nodes(inflow&outflow), finally nodes update by 1-FC
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device) # int64
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0) # edge_attr->shape (num_edge,2); edge_attr.mean(dim=0)->shape (2) ---broadcasting
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        e_h = 32
        e_out = 30
        n_out = out_dim
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   ) # 29->32->30
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   ) # 30->13

    def forward(self, x): # Size([batch_size, num_node, dim_feature])
        self.edge_index = self.edge_index.to(self.device) # ([2, num_edge])
        self.edge_attr = self.edge_attr.to(self.device) # ([num_edge, 2])

        edge_src, edge_target = self.edge_index # edge_index Size([2, num_edge])->edge_src, edge_target (num_edge,) respectively
        node_src = x[:, edge_src] # Size([batch_size, num_edge, dim_feature]) edge_src has index function
        node_target = x[:, edge_target]

        src_wind = node_src[:,:,-2:] * self.wind_std[None,None,:] + self.wind_mean[None,None,:] # denormalization; wind: speed+direc; wind_mean&wind_std->length:2 [batch_size, num_edge, 2] ---broadcasting
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:, :, 1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1) # add batch_size dimension
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]

        theta = torch.abs(city_direc - src_wind_direc) # to calc coefficient S(edge attr)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist) # 3 means data's time interval
        edge_weight = edge_weight.to(self.device)
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device) # add batch_size dimension
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:,:,None]], dim=-1) # total attr for each edge ([batch_size, num_edge, 29])

        out = self.edge_mlp(out) # channel:29->batch_size->30 Size([batch_size, num_edge, 30])
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1)) # Size([batch_size, num_edge, 30])->([batch_size, num_node, 30]) each cities' inflow; edge_target save the target node's index
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # For higher version of PyG. each cities' outflow

        out = out_add + out_sub # inflow + (-outflow)
        out = self.node_mlp(out) # 30->dim_feature

        return out


class MSGNN(nn.Module):
    def __init__(self, gpu_id, hist_len, pred_len, in_dim, out_dim, hid_dim, node_num, batch_size, edge_index, edge_attr, wind_mean, wind_std, use_met, use_time, interval_set, k_hop):
        super(MSGNN, self).__init__()

        self.device = 'cuda:%d' % gpu_id
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.node_num = node_num
        self.batch_size = batch_size
        self.use_met = use_met
        self.use_time = use_time

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.gnn_out = in_dim

        self.interval_set = interval_set
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        self.k_hop = k_hop
        self.fc_out = nn.Linear(self.hid_dim * self.k_hop , self.out_dim) # 64->1

        self.MSGRU = MSGRU(self.hist_len, self.pred_len, self.k_hop, self.batch_size, self.node_num, self.in_dim + self.gnn_out, self.hid_dim, self.interval_set, self.graph_gnn, self.fc_out, self.device, use_met, use_time)

    def forward(self, model_input): 
        y_pred = self.MSGRU(model_input)
        y_pred = torch.stack(y_pred, dim=1)

        return y_pred
