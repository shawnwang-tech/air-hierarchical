from models.cells import GRUCell, MTGRUCell
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj
import torch
from torch import nn


class MTCity_GC_GRU(nn.Module):
    def __init__(self, gpu_id, in_dim, out_dim, hid_dim, gnn_out_dim, batch_size, use_met, use_time, hist_len, pred_len, sta_num, city_num, adj_sta, adj_city, afc_s2c, interval_set, k_hop, norm):
        super(MTCity_GC_GRU, self).__init__()
        self.device = 'cuda:%d' % gpu_id
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.batch_size = batch_size
        self.use_met = use_met
        self.use_time = use_time
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.sta_num = sta_num
        self.city_num = city_num
        self.adj_sta = adj_sta
        self.adj_city = adj_city
        self.afc_s2c = afc_s2c

        # sta
        self.edge_index, self.edge_attr = dense_to_sparse(torch.FloatTensor(self.adj_sta))

        self.edge_index = self.edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1) * self.sta_num # 等价于batch_size张图, 共有batch_size*sta_num个节点, 不同batch间节点不相连
        self.edge_index = self.edge_index.view(2, -1)

        self.edge_attr = self.edge_attr.repeat(batch_size) 

        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)

        # city
        self.c_edge_index, self.c_edge_attr = dense_to_sparse(torch.FloatTensor(self.adj_city))

        self.c_edge_index = self.c_edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1) * self.city_num # 等价于batch_size张图, 共有batch_size*city_num个节点, 不同batch间节点不相连
        self.c_edge_index = self.c_edge_index.view(2, -1)

        self.c_edge_attr = self.c_edge_attr.repeat(batch_size) 

        self.c_edge_index = self.c_edge_index.to(self.device)
        self.c_edge_attr = self.c_edge_attr.to(self.device)

        self.gcn_out = gnn_out_dim

        self.norm = norm
        if self.norm == 'rw':
            self.conv = ChebConv(self.in_dim, self.gcn_out, K=k_hop, normalization='rw') # 实际K=K
            # 计算lambda_max
            data = Data(edge_index=self.edge_index, edge_attr=None, num_nodes=self.sta_num * batch_size)
            self.lambda_max = LaplacianLambdaMax()(data).lambda_max
        else:
            self.conv = ChebConv(self.in_dim, self.gcn_out, K=k_hop) # 实际K=K

        self.gru_cell = GRUCell(self.in_dim + self.gcn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        if self.norm == 'rw':
            self.c_conv = ChebConv(self.in_dim, self.gcn_out, K=k_hop, normalization='rw') # 实际K=K
            # 计算lambda_max
            c_data = Data(edge_index=self.c_edge_index, edge_attr=None, num_nodes=self.city_num * batch_size)
            self.c_lambda_max = LaplacianLambdaMax()(c_data).lambda_max
        else:
            self.c_conv = ChebConv(self.in_dim, self.gcn_out, K=k_hop) # 实际K=K
            
    
        self.gru_cell = MTGRUCell(self.in_dim + self.gcn_out, self.hid_dim, interval_set)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.c_gru_cell = MTGRUCell(self.in_dim + self.gcn_out, self.hid_dim, interval_set)
        self.c_fc_out = nn.Linear(self.hid_dim, self.out_dim)

        self.c2s = nn.Linear(gnn_out_dim, gnn_out_dim)
        self.s2c = nn.Linear(gnn_out_dim, gnn_out_dim)

    def forward(self, model_input):
        y_pred = []
        c_y_pred = []

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

        h0 = torch.zeros(self.batch_size * self.sta_num, self.hid_dim).to(self.device)
        hn = h0

        hc0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hcn = hc0

        idx = 1
        for t in range(1, self.hist_len):
            x = torch.cat([x_hist[:, t-1], features[:, t]], dim=-1)
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(self.batch_size * self.sta_num, -1)
            if self.norm == 'rw':
                x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index, lambda_max=self.lambda_max))
            else:
                x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index))
            x_gcn = x_gcn.view(self.batch_size, self.sta_num, -1)

            xc = torch.cat([c_x_hist[:, t-1], c_features[:, t]], dim=-1)
            xc_gcn = xc.contiguous()
            xc_gcn = xc_gcn.view(self.batch_size * self.city_num, -1)
            if self.norm == 'rw':
                xc_gcn = F.sigmoid(self.c_conv(xc_gcn, self.c_edge_index, lambda_max=self.c_lambda_max))
            else:
                xc_gcn = F.sigmoid(self.c_conv(xc_gcn, self.c_edge_index))
            xc_gcn = xc_gcn.view(self.batch_size, self.city_num, -1)

            x_gcn_tmp = x_gcn.clone()
            x_gcn = x_gcn + self.c2s(torch.bmm(torch.FloatTensor(self.afc_s2c[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), xc_gcn))
            xc_gcn = xc_gcn + self.s2c(torch.bmm(torch.FloatTensor(self.afc_s2c.transpose(1,0)[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), x_gcn_tmp))
            
            x_gru = torch.cat((x, x_gcn), dim=-1)
            xc_gru = torch.cat((xc, xc_gcn), dim=-1)

            hn = self.gru_cell(x_gru, hn, idx)
            hcn = self.c_gru_cell(xc_gru, hcn, idx)

            idx += 1

        x = x_hist[:, t]
        xc = c_x_hist[:, t]

        for i in range(self.pred_len):
            x = torch.cat([x, features[:, self.hist_len + i]], dim=-1)
            x_gcn = x.contiguous()
            x_gcn = x_gcn.view(self.batch_size * self.sta_num, -1)
            if self.norm == 'rw':
                x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index, lambda_max=self.lambda_max))
            else:
                x_gcn = F.sigmoid(self.conv(x_gcn, self.edge_index))
            x_gcn = x_gcn.view(self.batch_size, self.sta_num, -1)

            xc = torch.cat([xc, c_features[:, self.hist_len + i]], dim=-1)
            xc_gcn = xc.contiguous()
            xc_gcn = xc_gcn.view(self.batch_size * self.city_num, -1)
            if self.norm == 'rw':
                xc_gcn = F.sigmoid(self.c_conv(xc_gcn, self.c_edge_index, lambda_max=self.c_lambda_max))
            else:
                xc_gcn = F.sigmoid(self.c_conv(xc_gcn, self.c_edge_index))
            xc_gcn = xc_gcn.view(self.batch_size, self.city_num, -1)

            x_gcn_tmp = x_gcn.clone()
            x_gcn = x_gcn + self.c2s(torch.bmm(torch.FloatTensor(self.afc_s2c[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), xc_gcn))
            xc_gcn = xc_gcn + self.s2c(torch.bmm(torch.FloatTensor(self.afc_s2c.transpose(1,0)[None, :, :]).repeat(self.batch_size, 1, 1).to(self.device), x_gcn_tmp))

            x_gru = torch.cat((x, x_gcn), dim=-1)
            xc_gru = torch.cat((xc, xc_gcn), dim=-1)

            hn = self.gru_cell(x_gru, hn, idx)
            hcn = self.c_gru_cell(xc_gru, hcn, idx)

            x_fc = hn.view(self.batch_size, self.sta_num, self.hid_dim)
            xc_fc = hcn.view(self.batch_size, self.city_num, self.hid_dim)
            x = self.fc_out(x_fc)
            xc = self.c_fc_out(xc_fc)

            y_pred.append(x)
            c_y_pred.append(xc)

            idx += 1

        y_pred = torch.stack(y_pred, dim=1)
        c_y_pred = torch.stack(c_y_pred, dim=1)

        return y_pred, c_y_pred
        