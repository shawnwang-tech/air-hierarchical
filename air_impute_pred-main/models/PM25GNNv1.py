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


class PM25GNN(nn.Module):
    def __init__(self, gpu_id, hist_len, pred_len, in_dim, out_dim, hid_dim, node_num, batch_size, edge_index, edge_attr, wind_mean, wind_std, use_met, use_time):
        super(PM25GNN, self).__init__()

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

        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        # self.hist_graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)

        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        # self.hist_gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)

        self.fc_out = nn.Linear(self.hid_dim, self.out_dim) # 64->1

    def forward(self, model_input): 
        enc, dec = model_input
        pm25_hist = enc[0] # [batch_size, hist_len, N, 1]
        enc_misc = enc[1:]

        if self.use_met or self.use_time:
            dec = torch.cat(dec, dim=-1)
            enc_misc = torch.cat(enc_misc, dim=-1)

        feature = torch.cat([enc_misc, dec], dim=1) # [batch_size, hist_len+pred_len, N, dim_features]

        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.node_num, self.hid_dim).to(self.device)
        hn = h0
        # hist_len data to gru 
        for t in range(1, self.hist_len): # range从1开始取是避免pm25_hist[:,t]作为输入进入RNN cell两次
            xn = torch.cat((pm25_hist[:, t-1], feature[:, t]), dim=-1) # (batch_size, N, 12+1)
            xn_gnn = xn
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)
            x = torch.cat((xn_gnn, xn), dim=-1)

            hn = self.gru_cell(x, hn)

        # transition
        # xn = self.fc_out(hn) # 感觉不好
        xn = pm25_hist[:,-1] # 取hist最后一个时间

        # pred_len pred
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1) # get the corresponding time period's feature form total 25 periods (1st to 25th, no 0th---corresponding to the last 24 time periods to be predicted)

            xn_gnn = x # torch.Size([batch_size, num_node, dim_features])
            # xn_gnn = xn_gnn.contiguous() # has been contiguous
            # xn_gnn = self.graph_gnn(xn_gnn) # Size([batch_size, num_node, dim_features]) -> Size([batch_size, num_node, dim_features])
            x = torch.cat([xn_gnn, x], dim=-1) # Size([batch_size, num_node, dim_features + dim_out]) combine origin and after message passing

            hn = self.gru_cell(x, hn) # update hidden state(hn); Size([batch_size * num_node, num_hidden])
            xn = hn.view(self.batch_size, self.node_num, self.hid_dim) # Size([batch_size, num_node, num_hidden])
            xn = self.fc_out(xn) # hidden2out num_hidden->1(pm25)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1) # torch.Size([batch_size, pred_len, num_node, 1])

        return pm25_pred
