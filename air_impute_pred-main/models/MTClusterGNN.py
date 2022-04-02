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
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc

class SoftCluster(nn.Module):
    '''
    soft cluster:
        learnable weight: softmax operation for each row, sum of each row equals 1
    '''
    def __init__(self, num_cluster, num_orinode, num_dim, hist_len):
        super(SoftCluster, self).__init__()
        self.num_cluster = num_cluster
        self.num_orinode = num_orinode
        self.num_dim = num_dim
        
        self.row_W = nn.Parameter(torch.randn(num_cluster, num_dim))
        self.row_W.requires_grad = True

        self.fc = nn.Linear(hist_len, 1)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x, orinode_loc): # x.shape(batch_size, hist_len, num_orinode, num_dim)
        batch_size, hist_len, _, _ = x.shape
        afc_c2r_set = []
        for i in range(self.num_orinode):
            afc_row = torch.bmm(self.row_W[None,:].repeat(batch_size*hist_len, 1, 1), x.permute(0,1,3,2)[:, :, :, i][:, :, :, None].reshape(-1, 3, 1)) # (batch_size * hist_len, num_cluster, 1)
            afc_row = afc_row.view(batch_size, hist_len, afc_row.shape[-2], afc_row.shape[-1])
            afc_row = afc_row.permute(0,2,3,1).contiguous()
            afc_row = self.fc(afc_row)
            afc_c2r_set.append(self.softmax(afc_row.squeeze(-1))) 
        afc_c2r = torch.cat(afc_c2r_set, dim=-1) # (batch_size, num_cluster, num_orinode)

        # orinode_loc (num_orinode, 2)
        # r_loc = torch.bmm(afc_c2r, orinode_loc[None,].repeat(batch_size,1,1)) # (batch_size, num_cluster, 2)
        # adj = torch.ones(self.num_cluster, self.num_cluster).to(x.device)

        # edge_index, _ = dense_to_sparse(adj) # edge_index: (2, valid_edge_num)

        # direc_arr = []
        # dist_km_arr =[]
        # for i in range(edge_index.shape[1]): # traverse valid_edge_num
        #     src, dest = edge_index[0, i], edge_index[1, i]
        #     src_lat, src_lon = self.nodes.iloc[src, :]['lat'], self.nodes.iloc[src, :]['lon']
        #     dest_lat, dest_lon = self.nodes.iloc[dest, :]['lat'], self.nodes.iloc[dest, :]['lon']
        #     src_loc = (src_lat, src_lon)
        #     dest_loc = (dest_lat, dest_lon)
        #     dist_km = geodesic(src_loc, dest_loc).kilometers

        #     def posi_nega_fg(value):
        #         if abs(value) == 0:
        #             print("same lat/lon")
        #             return 0
        #         else:
        #             return value/abs(value)

        #     v = posi_nega_fg(src_lat - dest_lat) * geodesic((src_lat, dest_lon), (dest_lat, dest_lon)).meters # keep the longitude consistent
        #     u = posi_nega_fg(src_lon - dest_lon) * geodesic((src_lat, src_lon), (src_lat, dest_lon)).meters # Keep the latitude consistent

        #     u = u * units.meter / units.second
        #     v = v * units.meter / units.second
        #     direc = mpcalc.wind_direction(u, v)._magnitude # direction form source node to destination node
        #     direc_arr.append(direc)
        #     dist_km_arr.append(dist_km)

        # direc_arr = np.stack(direc_arr) 
        # dist_km_arr = np.stack(dist_km_arr)
        # attr = np.stack([dist_km_arr, direc_arr], axis=-1) # (valid_edge_num, 2)

        return afc_c2r

class MTGRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, interval_set, bias=True): # k: list->代表时间间隔
        super(MTGRUCell, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size *  len(interval_set)
        self.interval_set = interval_set
        self.num_interval = len(interval_set)
        self.hidden_slice_size = int(hidden_size / self.num_interval)
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.x2a = nn.Linear(input_size, self.num_interval, bias=bias)
        self.h2a = nn.Linear(hidden_size, self.num_interval, bias=bias)
        self.reset_params()
    
    def reset_params(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden, idx):
        x = x.reshape(-1, x.size(-1))

        xa = self.x2a(x) # 用于计算不同时间尺度的权重
        ha = self.h2a(hidden)

        a = torch.softmax(xa + ha, dim=-1)

        def weight_hidden(num_interval, hidden_slice_size, hidden, a): # 每个时间尺度按权重更新
            weight_hidden_list = []
            for i in range(num_interval):
                weight_hidden_list.append(hidden[:, hidden_slice_size*i : hidden_slice_size*(i+1)] * a[:, i][:, None].repeat(1,hidden_slice_size))
            return torch.cat(weight_hidden_list, dim=-1)

        isupdate_set = idx % np.array(self.interval_set) # 判断当前idx下是否更新
        weight_hidden = weight_hidden(self.num_interval, self.hidden_slice_size, hidden, a)

        gate_x = self.x2h(x)
        gate_h = self.h2h(weight_hidden)

        xz, xr, xh = gate_x.chunk(3, 1)
        hz, hr, hh = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(xr + hr)
        inputgate = torch.sigmoid(xz + hz)
        newgate = torch.tanh(xh + (resetgate * hh))

        update_all_hidden = newgate + inputgate * (weight_hidden - newgate)
        
        def mask_fill(isupdate_set, num_interval, hidden): # 根据isupdate_set保留hidden/更新
            new_hidden_list = []
            # hidden需分为k份考虑, 部分直接copy, 部分update
            for i, isupdate in enumerate(isupdate_set):
                if isupdate == 0:
                    new_hidden_list.append(update_all_hidden[:, num_interval*i : num_interval*(i+1)])
                else:
                    new_hidden_list.append(hidden[:, num_interval*i : num_interval*(i+1)])
            return torch.cat(new_hidden_list, dim=-1)

        new_hidden = mask_fill(isupdate_set, self.hidden_slice_size, hidden)

        return new_hidden

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

class CGraphGNN(nn.Module):
    def __init__(self, device, in_dim, out_dim, Nc):
        super(CGraphGNN, self).__init__()
        self.device = device
        self.Nc = Nc
        e_h = 32
        e_out = 30
        n_out = out_dim
        self.edge_mlp = Sequential(Linear((in_dim+1)* 2 + 6, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   ) # 29->32->30
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   ) # 30->13
        return
    
    def forward(self, x, adj_dist, adj_direct, wind_mean, wind_std):
        '''
        adj_dist/adj_direct: (32, 10, 10)->(32, 10, 10, 1)
        adj: (32, 10, 10, 1)
        x: (32, 10, 13)
        需要计算每条边的S分数
        shape: (32, 10, 10, 29)
        其中
            都新增一个维度13+1 1+1，新维度代表是否有边
            node_src: (32, 10, 10, 14)
            node_target: (32, 10, 10, 14)
            adj_dist_norm: (32, 10, 10, 2)
            adj_direct_nrom: (32, 10, 10, 2)
            adj_s: (32, 10, 10, 2)
        PS: 行是src, 列是dest
        '''
        # self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        # self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        self.adj_dist = adj_dist.unsqueeze(dim=-1)
        self.adj_direct = adj_direct.unsqueeze(dim=-1)
        ## node_src
        adj = (adj_dist != 0) # bool mask (32, 10, 10)
        adj = adj.type(torch.int).unsqueeze(dim=-1) # (32, 10, 10, 1) mask

        # 完整的node_src + mask得到带边的
        node_src_full = x.unsqueeze(dim=2).repeat(1,1,self.Nc,1) # (32, 10, 10, 13)
        node_target_full = x.unsqueeze(dim=1).repeat(1,self.Nc,1,1)

        node_src = adj.repeat(1,1,1,12) * node_src_full
        node_target = adj.repeat(1,1,1,12) * node_target_full

        self.node_src = torch.cat((node_src, adj), dim=-1)
        self.node_target = torch.cat((node_target, adj), dim=-1)

        # norm
        self.adj_dist_norm = (self.adj_dist - self.adj_dist.mean(dim=(0,1))) / self.adj_dist.std(dim=(0,1))
        self.adj_direct_norm = (self.adj_direct - self.adj_direct.mean(dim=(0,1))) / self.adj_direct.std(dim=(0,1))
    
        ## calculate S
        src_wind = self.node_src[:,:,:,2:4] * self.wind_std[:, None, None, :] + self.wind_mean[:, None, None, :] 
        src_wind_speed = src_wind[:, :, :, 0]
        src_wind_direc = src_wind[:, :, :, 1]
        
        theta = torch.abs(adj_direct - src_wind_direc)
        adj_weight = torch.where(adj_dist==0, torch.zeros_like(theta, dtype=torch.float32), F.relu(3 * src_wind_speed * torch.cos(theta) / adj_dist)).unsqueeze(dim=-1)
        adj_weight_0 = (F.relu(3 * src_wind_speed * torch.cos(theta) / adj_dist)).unsqueeze(dim=-1)
        # # adj_inverse
        # adj_inverse_temp = torch.ones_like(adj, dtype=torch.float32)
        # adj_inverse = adj_inverse_temp - adj

        adj_weight = adj_weight.to(self.device)

        out0 = torch.cat([self.node_src, self.node_target,\
                        torch.cat([self.adj_dist_norm, adj], dim=-1),\
                        torch.cat([self.adj_direct_norm, adj],dim=-1),\
                        torch.cat([adj, adj], dim=-1)\
                        ], dim=-1).type(torch.float)

        out = self.edge_mlp(out0) # (32, 10, 10, 30)
        # -> (32, 10, 30) inflow
        out_add = out.sum(dim=1)
        # -> (32, 10, 30) outflow2
        out_sub = out.neg().sum(dim=2)

        out = out_add + out_sub
        out = self.node_mlp(out)

        return out

class MTMSGRU(nn.Module):
    
    def __init__(self, hist_len, pred_len, k_hop, batch_size, num_node, city_num, 
                 input_dim, hid_dim, interval_set, 
                 graph_gnn, c_graph_gnn, r_graph_gnn, 
                 fc_out, c_fc_out, r_fc_out, c2s, s2c, r2c, afc_s2c,
                 device,
                 use_met, use_time,
                 city_info_norm, city_info, num_cluster):
                 
        super(MTMSGRU, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.k_hop = k_hop
        self.batch_size = batch_size 
        self.num_node = num_node
        self.city_num = city_num
        self.hid_dim = hid_dim
        self.num_cluster = num_cluster
        self.city_info_norm = city_info_norm
        self.city_info = city_info

        self.use_met = use_met
        self.use_time = use_time

        self.graph_gnn = graph_gnn
        self.c_graph_gnn = c_graph_gnn
        self.r_graph_gnn = r_graph_gnn

        self.fc_out = fc_out
        self.c_fc_out = c_fc_out
        self.r_fc_out = r_fc_out
        self.c2s = c2s
        self.s2c = s2c
        self.r2c = r2c

        self.afc_s2c = afc_s2c

        self.device = device
        
        self.mtgru_cell_set = nn.ModuleList([MTGRUCell(input_dim, hid_dim, interval_set) for _ in range(k_hop)])
        self.c_mtgru_cell_set = nn.ModuleList([MTGRUCell(input_dim, hid_dim, interval_set) for _ in range(k_hop)])
        self.r_mtgru_cell_set = nn.ModuleList([MTGRUCell(input_dim, hid_dim, interval_set) for _ in range(k_hop)])

        self.soft_cluster = SoftCluster(self.num_cluster, city_num, 3, self.hist_len)
        self.dist_adj_fc = nn.Linear(self.city_num, self.num_cluster)
        self.direc_adj_fc = nn.Linear(self.city_num, self.num_cluster)
 
    def forward(self, model_input):
        '''
        1. 
            hist_len中x直接从x_hist获取
            在pred_len中, 需要k_hop结合预测得下一时刻的x值    
        '''
        def get_from_input(model_input_i):
            enc, dec = model_input_i
            x_hist = enc[0] # [batch_size, hist_len, N, 1]
            enc_misc = enc[1:]

            if self.use_met or self.use_time:
                dec = torch.cat(dec, dim=-1)
                enc_misc = torch.cat(enc_misc, dim=-1)

            features = torch.cat([enc_misc, dec], dim=1) # [batch_size, hist_len+pred_len, N, dim_features]
            return x_hist, features
    
        x_hist, features = get_from_input(model_input[0])
        c_x_hist, c_features = get_from_input(model_input[1])

        ### soft cluster
        city_ori_info = torch.cat([self.city_info_norm[None, None, :, :].repeat(self.batch_size, self.hist_len, 1, 1), \
                                  c_x_hist], dim=-1) # (batch_size, hist_len, num_orinode, 3)
        afc_c2r = self.soft_cluster(city_ori_info, self.city_info) # (batch_size, num_cluster, num_orinode)
        # get r_hist, r_features
        r_x_hist = torch.bmm(afc_c2r.repeat(self.hist_len,1,1), c_x_hist.view(self.batch_size*self.hist_len, self.city_num, -1)).view(self.batch_size, self.hist_len, self.num_cluster, -1)
        self.seq_len = self.hist_len + self.pred_len
        r_features = torch.bmm(afc_c2r.repeat(self.seq_len,1,1), c_features.view(self.batch_size*self.seq_len, self.city_num, -1)).view(self.batch_size, self.seq_len, self.num_cluster, -1)

        r_wind_mean = r_features.mean(dim = [0,1,2])[1:3]
        r_wind_std = r_features.std(dim = [0,1,2])[1:3]
        dist_adj = self.dist_adj_fc(afc_c2r)
        direc_adj = self.direc_adj_fc(afc_c2r)
    
        y_pred = []
        c_y_pred = []
        
        h0_set = []
        hc0_set = []
        hr0_set = []
        for k in range(self.k_hop): # 初始化hidden state
            h0_set.append(torch.zeros(self.batch_size * self.num_node, self.hid_dim).to(self.device))
            hc0_set.append(torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device))
            hr0_set.append(torch.zeros(self.batch_size * self.num_cluster, self.hid_dim).to(self.device))

        hn_set = h0_set
        hcn_set = hc0_set
        hrn_set = hr0_set
        idx = 1 # update idx start from 1

        for t in range(self.hist_len-1):
            xn = torch.cat([x_hist[:, t], features[:, t]], dim=-1)
            xn_gnn = xn
            xn_gnn = xn_gnn.contiguous()

            xcn = torch.cat([c_x_hist[:, t], c_features[:, t]], dim=-1)
            xcn_gnn = xcn
            xcn_gnn = xcn_gnn.contiguous()

            xrn = torch.cat([r_x_hist[:, t], r_features[:, t]], dim=-1)
            xrn_gnn = xrn
            xrn_gnn = xrn_gnn.contiguous()

            for k in range(self.k_hop):
                xn_gnn = self.graph_gnn(xn_gnn)            
                xcn_gnn = self.c_graph_gnn(xcn_gnn)
                xrn_gnn = self.r_graph_gnn(xrn_gnn, dist_adj, direc_adj, r_wind_mean[None,].repeat(self.batch_size, 1), r_wind_std[None,].repeat(self.batch_size, 1))

                xn_gnn_tmp = xn_gnn.clone()
                xcn_gnn = xcn_gnn + self.r2c(torch.bmm(afc_c2r.permute(0,2,1), xrn_gnn))
                xn_gnn = xn_gnn + self.c2s(torch.bmm(torch.FloatTensor(self.afc_s2c[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), xcn_gnn))
                xcn_gnn = xcn_gnn + self.s2c(torch.bmm(torch.FloatTensor(self.afc_s2c.transpose(1,0)[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), xn_gnn_tmp))
               
                x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1) # clone() -> Gradient Accumulate
                hn_set[k] = self.mtgru_cell_set[k](x_gru, hn_set[k], idx)

                xc_gru = torch.cat([xcn_gnn, xcn.clone()], dim=-1) # clone() -> Gradient Accumulate
                hcn_set[k] = self.c_mtgru_cell_set[k](xc_gru, hcn_set[k], idx)
                
                xr_gru = torch.cat([xrn_gnn, xrn.clone()], dim=-1) # clone() -> Gradient Accumulate
                hrn_set[k] = self.r_mtgru_cell_set[k](xr_gru, hrn_set[k], idx)

                idx += 1

        x = x_hist[:, t+1]
        xc = c_x_hist[:, t+1]
        xr = r_x_hist[:, t+1]

        for i in range(self.pred_len):
            xn = torch.cat([x, features[:, self.hist_len - 1 + i]], dim=-1)
            xn_gnn = xn
            xn_gnn = xn_gnn.contiguous()

            xcn = torch.cat([xc, c_features[:, self.hist_len - 1 + i]], dim=-1)
            xcn_gnn = xcn
            xcn_gnn = xcn_gnn.contiguous()

            xrn = torch.cat([xr, r_features[:, self.hist_len - 1 + i]], dim=-1)
            xrn_gnn = xrn
            xrn_gnn = xrn_gnn.contiguous()

            for k in range(self.k_hop):
                xn_gnn = self.graph_gnn(xn_gnn)
                xcn_gnn = self.c_graph_gnn(xcn_gnn)
                xrn_gnn = self.r_graph_gnn(xrn_gnn, dist_adj, direc_adj, r_wind_mean[None,].repeat(self.batch_size, 1), r_wind_std[None,].repeat(self.batch_size, 1))

                xn_gnn_tmp = xn_gnn.clone()
                xcn_gnn = xcn_gnn + self.r2c(torch.bmm(afc_c2r.permute(0,2,1), xrn_gnn))
                xn_gnn = xn_gnn + self.c2s(torch.bmm(torch.FloatTensor(self.afc_s2c[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), xcn_gnn))
                xcn_gnn = xcn_gnn + self.s2c(torch.bmm(torch.FloatTensor(self.afc_s2c.transpose(1,0)[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), xn_gnn_tmp))

                x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
                hn_set[k] = self.mtgru_cell_set[k](x_gru, hn_set[k], idx)

                xc_gru = torch.cat([xcn_gnn, xcn.clone()], dim=-1) # clone() -> Gradient Accumulate
                hcn_set[k] = self.c_mtgru_cell_set[k](xc_gru, hcn_set[k], idx)

                xr_gru = torch.cat([xrn_gnn, xrn.clone()], dim=-1) # clone() -> Gradient Accumulate
                hrn_set[k] = self.r_mtgru_cell_set[k](xr_gru, hrn_set[k], idx)
            
            hn_all_set = []
            hcn_all_set = []
            hrn_all_set = []
            for k in range(self.k_hop):
                hn_all_set.append(hn_set[k].view(self.batch_size, self.num_node, self.hid_dim))
                hcn_all_set.append(hcn_set[k].view(self.batch_size, self.city_num, self.hid_dim))
                hrn_all_set.append(hrn_set[k].view(self.batch_size, self.num_cluster, self.hid_dim))
            
            hn_all = torch.cat(hn_all_set, dim=-1) # 合并不同hop gcn后的结果
            hcn_all = torch.cat(hcn_all_set, dim=-1)
            hrn_all = torch.cat(hrn_all_set, dim=-1)
            x = self.fc_out(hn_all)
            xc = self.c_fc_out(hcn_all)
            xr = self.r_fc_out(hrn_all)
            y_pred.append(x)
            c_y_pred.append(xc)

            idx += 1

        return y_pred, c_y_pred

class MTClusterGNN(nn.Module):
    def __init__(self, gpu_id, hist_len, pred_len, in_dim, out_dim, hid_dim, node_num, city_num, afc_s2c, batch_size, edge_index, edge_attr, c_edge_index, c_edge_attr, wind_mean, wind_std, c_wind_mean, c_wind_std, use_met, use_time, interval_set, k_hop, city_info_norm, city_info, num_cluster):
        super(MTClusterGNN, self).__init__()

        self.device = 'cuda:%d' % gpu_id
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.node_num = node_num
        self.city_num = city_num
        self.afc_s2c = afc_s2c
        self.batch_size = batch_size
        self.use_met = use_met
        self.use_time = use_time

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.gnn_out = in_dim

        self.city_info_norm = torch.Tensor(np.float32(city_info_norm)).to(self.device)
        self.city_info = torch.Tensor(np.float32(city_info)).to(self.device)

        self.num_cluster = num_cluster
        self.interval_set = interval_set    
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        self.c_graph_gnn = GraphGNN(self.device, c_edge_index, c_edge_attr, self.in_dim, self.gnn_out, c_wind_mean, c_wind_std)
        self.r_graph_gnn = CGraphGNN(self.device, self.in_dim, self.gnn_out, self.num_cluster)

        self.k_hop = k_hop
        self.fc_out = nn.Linear(self.hid_dim * self.k_hop , self.out_dim) 
        self.c_fc_out = nn.Linear(self.hid_dim * self.k_hop , self.out_dim) 
        self.r_fc_out = nn.Linear(self.hid_dim * self.k_hop , self.out_dim) 
        self.c2s = nn.Linear(in_dim, in_dim)
        self.s2c = nn.Linear(in_dim, in_dim)
        self.r2c = nn.Linear(in_dim, in_dim)

        self.MTMSGRU = MTMSGRU(self.hist_len, self.pred_len, self.k_hop, self.batch_size, self.node_num, self.city_num, self.in_dim + self.gnn_out, self.hid_dim, self.interval_set, self.graph_gnn, self.c_graph_gnn, self.r_graph_gnn, self.fc_out, self.c_fc_out, self.r_fc_out, self.c2s, self.s2c, self.r2c, self.afc_s2c, self.device, use_met, use_time, self.city_info_norm, self.city_info, self.num_cluster)

    def forward(self, model_input):
        y_pred, c_y_pred = self.MTMSGRU(model_input)

        y_pred = torch.stack(y_pred, dim=1)
        c_y_pred = torch.stack(c_y_pred, dim=1)

        return y_pred, c_y_pred

