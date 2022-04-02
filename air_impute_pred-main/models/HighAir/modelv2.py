from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add

from models.cells import GRUCell, LSTMCell


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

    def forward(self, x, h0, c0):
        # Set initial hidden and cell states
        # Forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        return h_n, c_n

class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.lin = nn.Linear(hidden_size, 1)
        # self.relu = nn.ReLU()

    def forward(self, x, h_0, c_0):
        # Forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.lin(out)
        # out = self.relu(out)
        # Decode the hidden state of the last time step
        return out, h_n, c_n

class CityModelv2(nn.Module):
    """Station graph"""
    def __init__(self, aqi_em, poi_em, wea_em, rnn_h, rnn_l, gnn_h, hist_len, pred_len, fc):
        super(CityModelv2, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.rnn_h = rnn_h
        self.gnn_h = gnn_h
        self.rnn_l = rnn_l

        # self.fc = fc
        # self.fc = nn.Linear(8, 24)
        self.fc = nn.Linear(8*16, 24)

        self.aqi_embed = Seq(Lin(1, aqi_em), ReLU())
        self.city_embed = Seq(Lin(gnn_h, wea_em), ReLU())
        self.wea_embed = Seq(Lin(11, wea_em), ReLU())
        # self.sta_gnn = StaGNN(aqi_em, 2, gnn_h, 2 * wea_em)
        # self.encoder = RNNEncoder(input_size=wea_em + aqi_em,
        #                           hidden_size=rnn_h,
        #                           num_layers=rnn_l)
        # self.decoder_embed = Seq(Lin(1, aqi_em), ReLU())
        # self.decoder = RNNDecoder(input_size=11 + aqi_em,
        #                           hidden_size=rnn_h,
        #                           num_layers=rnn_l)
        
        # self.encoder = RNNEncoder(input_size=12,
        #                           hidden_size=rnn_h,
        #                           num_layers=rnn_l)
        # self.decoder_embed = Seq(Lin(1, aqi_em), ReLU())
        # self.decoder = RNNDecoder(input_size=12,
        #                           hidden_size=rnn_h,
        #                           num_layers=rnn_l)

        self.gru = GRUCell(12, 32)
        # self.lstm = LSTMCell(12, 32)
        self.fc_out = nn.Linear(32, 1)

        self.lstm_en = nn.LSTM(12,
                            32,
                            1,
                            batch_first=True)
        
        self.lstm_de = nn.LSTM(12,
                            32,
                            1,
                            batch_first=True)

        self.encoder = RNNEncoder(input_size=gnn_h,
                                  hidden_size=32,
                                  num_layers=1)

        self.decoder = RNNDecoder(input_size=12,
                                  hidden_size=32,
                                  num_layers=1)

        self.sta_gnn = StaGNN(1, 2, gnn_h, 11+32)

    def batchInput(self, x, edge_w, edge_conn):
        sta_num = x.shape[1]
        x = x.reshape(-1, x.shape[-1])
        edge_w = edge_w.reshape(-1, edge_w.shape[-1])
        for i in range(edge_conn.size(0)):
            edge_conn[i, :] = torch.add(edge_conn[i, :], i * sta_num)
        edge_conn = edge_conn.transpose(0, 1)
        edge_conn = edge_conn.reshape(2, -1)
        return x, edge_w, edge_conn

    def forward(self, city_data, city_u, device):
        
        ### 上city u
        y_pred = []

        sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        sta_num = sta_aqi.shape[1]
        sta_x = sta_aqi
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)
        
        h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        h0 = h0[None,]
        c0 = c0[None,]

        hn = h0
        cn = c0
        idx = 1 # update idx start from 1

        sta_x = sta_x.reshape(-1, sta_x.shape[-2], sta_x.shape[-1])
        sta_conn = sta_conn.transpose(1, 2).repeat(self.hist_len, 1, 1) # (batch_size*hist_len, 2, conn_num)
        sta_w = sta_w.reshape(-1, sta_w.shape[-2], sta_w.shape[-1]) # (batch_size*hist_len, conn_num, fea_dim)

        sta_x, sta_weight, sta_conn = self.batchInput(sta_x, sta_w, sta_conn)

        sta_u = torch.cat([city_u, sta_wea], dim=-1) # city->sta: global feature u

        sta_x = self.sta_gnn(sta_x, sta_conn, sta_weight, sta_u, sta_num)
        sta_x = sta_x.reshape(-1, self.hist_len, sta_num, sta_x.shape[-1]).transpose(1, 2)
        sta_x = sta_x.reshape(-1, self.hist_len, sta_x.shape[-1])

        # x_hist = torch.cat([sta_aqi, sta_wea], dim=-1).permute(0,2,1,3)
        # x_hist = x_hist.reshape(-1, x_hist.shape[-2], x_hist.shape[-1])

        hn, cn = self.encoder(sta_x, hn, cn)

        x = sta_aqi[:, :, -1]

        for i in range(self.pred_len):
            xn = torch.cat([x, sta_for[:, i]], dim=-1)

            xn_gnn = xn
            xn_gnn = xn_gnn.contiguous()

            # xn_gnn = self.graph_gnn(xn_gnn)
            # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
            # hn = self.gru(xn_gnn, hn)
            # hn, cn = self.lstm(xn_gnn, (hn, cn))
            xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
            out, hn, cn = self.decoder(xn_gnn, hn, cn)
            x = out.squeeze(axis=-1).reshape(64, sta_num, -1)
            
            y_pred.append(x)

            idx += 1

        y_pred = torch.cat(y_pred, dim=-1)
        outputs = y_pred

        ### 上gnn
        # y_pred = []

        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_x = sta_aqi
        # # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # h0 = h0[None,]
        # c0 = c0[None,]

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # sta_x = sta_x.reshape(-1, sta_x.shape[-2], sta_x.shape[-1])
        # sta_conn = sta_conn.transpose(1, 2).repeat(self.hist_len, 1, 1) # (batch_size*hist_len, 2, conn_num)
        # sta_w = sta_w.reshape(-1, sta_w.shape[-2], sta_w.shape[-1]) # (batch_size*hist_len, conn_num, fea_dim)

        # sta_x, sta_weight, sta_conn = self.batchInput(sta_x, sta_w, sta_conn)
        # # city_u = self.city_embed(city_u)
        # # sta_wea = self.wea_embed(sta_wea)
        # sta_u = torch.cat([city_u, sta_wea], dim=-1) # city->sta: global feature u

        # # sta_u = torch.cat([sta_wea], dim=-1) # city->sta: global feature u
        # sta_x = self.sta_gnn(sta_x, sta_conn, sta_weight, sta_u, sta_num)
        # sta_x = sta_x.reshape(-1, self.hist_len, sta_num, sta_x.shape[-1]).transpose(1, 2)
        # sta_x = sta_x.reshape(-1, self.hist_len, sta_x.shape[-1])

        # # x_hist = torch.cat([sta_aqi, sta_wea], dim=-1).permute(0,2,1,3)
        # # x_hist = x_hist.reshape(-1, x_hist.shape[-2], x_hist.shape[-1])

        # hn, cn = self.encoder(sta_x, hn, cn)

        # x = sta_aqi[:, :, -1]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, hn, cn = self.decoder(xn_gnn, hn, cn)
        #     x = out.squeeze(axis=-1).reshape(64, sta_num, -1)
            
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred


        ### 比较下lstm hist和pred不是同一个 在1次性的基础上改
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_aqi = sta_aqi.permute(0,2,1,3)
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)

        # y_pred = []
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # h0 = h0[None,]
        # c0 = c0[None,]

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # # for t in range(self.hist_len):
        # #     xn = torch.cat([sta_aqi[:, t], sta_wea[:, t]], dim=-1)
        # #     xn_gnn = xn
        # #     xn_gnn = xn_gnn.contiguous()

        # #     # xn_gnn = self.graph_gnn(xn_gnn)
        # #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1) # clone() -> Gradient Accumulate
        # #     # hn= self.gru(xn_gnn, (hn))
        # #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        # #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        # #     out, (hn, cn) = self.lstm(xn_gnn, (hn, cn))
            
        # #     idx += 1

        # x_hist = torch.cat([sta_aqi, sta_wea], dim=-1).permute(0,2,1,3)
        # x_hist = x_hist.reshape(-1, x_hist.shape[-2], x_hist.shape[-1])
        # out, (hn, cn) = self.lstm_de(x_hist, (hn, cn))

        # x = sta_aqi[:, -1]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, (hn, cn) = self.lstm_de(xn_gnn, (hn, cn))
            
        #     hn_out = hn.view(64, sta_num, 32)
            
        #     x = self.fc_out(hn_out)
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred

        ### 上decoder
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_aqi = sta_aqi.permute(0,2,1,3)
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)

        # y_pred = []
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # h0 = h0[None,]
        # c0 = c0[None,]

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # x_hist = torch.cat([sta_aqi, sta_wea], dim=-1).permute(0,2,1,3)
        # x_hist = x_hist.reshape(-1, x_hist.shape[-2], x_hist.shape[-1])
        # hn, cn = self.encoder(x_hist, hn, cn)

        # x = sta_aqi[:, -1]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, hn, cn = self.decoder(xn_gnn, hn, cn)
        #     x = out.squeeze(axis=-1).reshape(64, sta_num, -1)
            
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred


        ### 上encoder
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_aqi = sta_aqi.permute(0,2,1,3)
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)

        # y_pred = []
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # h0 = h0[None,]
        # c0 = c0[None,]

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # x_hist = torch.cat([sta_aqi, sta_wea], dim=-1).permute(0,2,1,3)
        # x_hist = x_hist.reshape(-1, x_hist.shape[-2], x_hist.shape[-1])
        # hn, cn = self.encoder(x_hist, hn, cn)

        # x = sta_aqi[:, -1]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, (hn, cn) = self.lstm(xn_gnn, (hn, cn))
            
        #     hn_out = hn.view(64, sta_num, 32)
            
        #     x = self.fc_out(hn_out)
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred

        ### torch的lstm hist改成一次性
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_aqi = sta_aqi.permute(0,2,1,3)
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)

        # y_pred = []
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # h0 = h0[None,]
        # c0 = c0[None,]

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # # for t in range(self.hist_len):
        # #     xn = torch.cat([sta_aqi[:, t], sta_wea[:, t]], dim=-1)
        # #     xn_gnn = xn
        # #     xn_gnn = xn_gnn.contiguous()

        # #     # xn_gnn = self.graph_gnn(xn_gnn)
        # #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1) # clone() -> Gradient Accumulate
        # #     # hn= self.gru(xn_gnn, (hn))
        # #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        # #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        # #     out, (hn, cn) = self.lstm(xn_gnn, (hn, cn))
            
        # #     idx += 1

        # x_hist = torch.cat([sta_aqi, sta_wea], dim=-1).permute(0,2,1,3)
        # x_hist = x_hist.reshape(-1, x_hist.shape[-2], x_hist.shape[-1])
        # out, (hn, cn) = self.lstm(x_hist, (hn, cn))

        # x = sta_aqi[:, -1]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, (hn, cn) = self.lstm(xn_gnn, (hn, cn))
            
        #     hn_out = hn.view(64, sta_num, 32)
            
        #     x = self.fc_out(hn_out)
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred


        ### torch的lstm
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_aqi = sta_aqi.permute(0,2,1,3)
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)

        # y_pred = []
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # h0 = h0[None,]
        # c0 = c0[None,]

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # for t in range(self.hist_len):
        #     xn = torch.cat([sta_aqi[:, t], sta_wea[:, t]], dim=-1)
        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1) # clone() -> Gradient Accumulate
        #     # hn= self.gru(xn_gnn, (hn))
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, (hn, cn) = self.lstm(xn_gnn, (hn, cn))
            
        #     idx += 1

        # x = sta_aqi[:, t]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     # hn, cn = self.lstm(xn_gnn, (hn, cn))
        #     xn_gnn = xn_gnn.reshape(-1, xn_gnn.shape[-1])[:,None]
        #     out, (hn, cn) = self.lstm(xn_gnn, (hn, cn))
            
        #     hn_out = hn.view(64, sta_num, 32)
            
        #     x = self.fc_out(hn_out)
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred

        ### 自己写的垃圾版 编解码 有用...
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data

        # sta_num = sta_aqi.shape[1]
        # sta_aqi = sta_aqi.permute(0,2,1,3)
        # sta_wea = sta_wea[:,:,None,:].repeat(1,1,sta_num,1)
        # sta_for = sta_for[:,:,None,:].repeat(1,1,sta_num,1)

        # y_pred = []
        
        # h0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)
        # c0 = torch.randn(64 * sta_num, 32).type_as(sta_aqi)

        # hn = h0
        # cn = c0
        # idx = 1 # update idx start from 1

        # for t in range(self.hist_len):
        #     xn = torch.cat([sta_aqi[:, t], sta_wea[:, t]], dim=-1)
        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1) # clone() -> Gradient Accumulate
        #     # hn= self.gru(xn_gnn, (hn))
        #     hn, cn = self.lstm(xn_gnn, (hn, cn))
            
        #     idx += 1

        # x = sta_aqi[:, t]

        # for i in range(self.pred_len):
        #     xn = torch.cat([x, sta_for[:, i]], dim=-1)

        #     xn_gnn = xn
        #     xn_gnn = xn_gnn.contiguous()

        #     # xn_gnn = self.graph_gnn(xn_gnn)
        #     # x_gru = torch.cat([xn_gnn, xn.clone()], dim=-1)
        #     # hn = self.gru(xn_gnn, hn)
        #     hn, cn = self.lstm(xn_gnn, (hn, cn))
            
        #     hn_out = hn.view(64, sta_num, 32)
            
        #     x = self.fc_out(hn_out)
        #     y_pred.append(x)

        #     idx += 1

        # y_pred = torch.cat(y_pred, dim=-1)
        # outputs = y_pred

        ### embed的编解码
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data
        # sta_num = sta_aqi.shape[1]
        # sta_x = self.aqi_embed(sta_aqi)
        # sta_x = sta_x.reshape(-1, sta_x.shape[-2], sta_x.shape[-1]) # (batch_size*sta_num, hist_len, fea_dim)

        # sta_wea = self.wea_embed(sta_wea)
        # sta_wea = sta_wea[:, None].repeat(1, sta_num, 1, 1)
        # sta_wea = sta_wea.reshape(-1, sta_wea.shape[-2], sta_wea.shape[-1]) # (batch_size*sta_num, hist_len, fea_dim)

        # sta_x = torch.cat([sta_x, sta_wea], dim=-1) # (batch_size*sta_num, hist_len, fea_dim)

        # h0 = torch.randn(self.rnn_l, sta_x.size(0), self.rnn_h).to(device) # (rnn_l, batch_size*sta_num, rnn_h_dim) batch_size*sta_num个lstm模型
        # c0 = torch.randn(self.rnn_l, sta_x.size(0), self.rnn_h).to(device)
        # h_x, c_x = self.encoder(sta_x, h0, c0)

        # outputs = torch.zeros((sta_x.size(0), sta_for.size(1), 1)).to(device) # (batch_size*sta_num, pred_len, 1)
        # aqi = sta_aqi[:, :, -1].reshape(-1, 1) # get last slot in hist_len
        # sta_for = sta_for.repeat(sta_num, 1, 1) # (batch_size*sta_num, pred_len, for_dim)
        # for i in range(sta_for.size(1)):
        #     aqi_em = self.decoder_embed(aqi)
        #     inputs = torch.cat((aqi_em, sta_for[:, i]), dim=-1)
        #     inputs = inputs.unsqueeze(dim=1)
        #     output, h_x, c_x = self.decoder(inputs, h_x, c_x)
        #     output = output.reshape(-1, 1)
        #     outputs[:, i] = output
        #     aqi = output
        # outputs = outputs.reshape(-1, sta_num, sta_for.size(1))

        ### fc
        # x_hist, sta_conn, sta_w, sta_wea, sta_for = city_data
        # x_hist = x_hist.squeeze()
        # y_pred = self.fc(x_hist)
        # y_pred = y_pred
        # outputs = y_pred

        ### embed fc
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data
        # sta_num = sta_aqi.shape[1]
        # sta_x = self.aqi_embed(sta_aqi)
        # sta_x = sta_x.reshape(sta_x.shape[0], sta_x.shape[1], -1)
        # outputs = self.fc(sta_x)

        ### 无embed的编解码
        # sta_aqi, sta_conn, sta_w, sta_wea, sta_for = city_data
        # sta_num = sta_aqi.shape[1]
        # sta_x = sta_aqi
        # sta_x = sta_x.reshape(-1, sta_x.shape[-2], sta_x.shape[-1]) # (batch_size*sta_num, hist_len, fea_dim)

        # sta_wea = sta_wea[:, None].repeat(1, sta_num, 1, 1)
        # sta_wea = sta_wea.reshape(-1, sta_wea.shape[-2], sta_wea.shape[-1]) # (batch_size*sta_num, hist_len, fea_dim)

        # sta_x = torch.cat([sta_x, sta_wea], dim=-1) # (batch_size*sta_num, hist_len, fea_dim) 12

        # h0 = torch.randn(self.rnn_l, sta_x.size(0), self.rnn_h).to(device) # (rnn_l, batch_size*sta_num, rnn_h_dim) batch_size*sta_num个lstm模型
        # c0 = torch.randn(self.rnn_l, sta_x.size(0), self.rnn_h).to(device)
        # h_x, c_x = self.encoder(sta_x, h0, c0)

        # outputs = torch.zeros((sta_x.size(0), sta_for.size(1), 1)).to(device) # (batch_size*sta_num, pred_len, 1)
        # aqi = sta_aqi[:, :, -1].reshape(-1, 1) # get last slot in hist_len
        # sta_for = sta_for.repeat(sta_num, 1, 1) # (batch_size*sta_num, pred_len, for_dim)
        # for i in range(sta_for.size(1)):
        #     aqi_em = aqi
        #     # aqi_em = self.decoder_embed(aqi)
        #     inputs = torch.cat((aqi_em, sta_for[:, i]), dim=-1)
        #     inputs = inputs.unsqueeze(dim=1)
        #     output, h_x, c_x = self.decoder(inputs, h_x, c_x)
        #     output = output.reshape(-1, 1)
        #     outputs[:, i] = output
        #     aqi = output
        # outputs = outputs.reshape(-1, sta_num, sta_for.size(1))

        return outputs
        

class StaGNN(torch.nn.Module):
    def __init__(self, node_h, edge_h, gnn_h, u_h):
        super(StaGNN, self).__init__()
        self.node_mlp_1 = Seq(Lin(2 * node_h + edge_h, gnn_h))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h + u_h, gnn_h))

    def forward(self, x, edge_index, edge_attr, u, sta_num):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        u = u.reshape(-1, u.shape[-1])
        u = u.repeat(sta_num, 1)
        row, col = edge_index
        out = torch.cat([x[row], x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out) # propagation
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u], dim=1) # global u
        return self.node_mlp_2(out) # aggregation