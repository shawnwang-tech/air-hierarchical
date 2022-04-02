import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from models.HighAir.modelv3 import CityModel, GlobalModel
from models.HighAir.modelv2 import CityModelv2

class HighAir(nn.Module):

    def __init__(self, gpu_id, aqi_em, poi_em, wea_em, rnn_h, rnn_l, gnn_h, city_num, sta_num, 
                 city_edge_index, city_edge_attr, edge_index_set, edge_attr_set, use_met, use_time, 
                 hist_len, pred_len, afc_sparse, valid_city_idx):
        super(HighAir, self).__init__()
        self.device = "cuda:%d" % gpu_id
        self.aqi_em = aqi_em
        self.poi_em = poi_em
        self.wea_em = wea_em
        self.rnn_h = rnn_h
        self.rnn_l = rnn_l
        self.gnn_h = gnn_h
        self.city_num = city_num
        self.sta_num = sta_num
        self.use_met = use_met
        self.use_time = use_time
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.afc_sparse = afc_sparse
        self.valid_city_idx = valid_city_idx
        self.valid_num = len(self.valid_city_idx)

        self.city_edge_index = torch.LongTensor(city_edge_index).to(self.device) # int64
        self.city_edge_attr = torch.Tensor(np.float32(city_edge_attr)).to(self.device)

        self.edge_index_set = edge_index_set
        self.edge_attr_set = edge_attr_set

        self.global_model = GlobalModel(self.aqi_em, self.rnn_h, self.rnn_l,
                                        self.gnn_h, self.city_num, 
                                        self.hist_len, self.pred_len).to(self.device)
                                                
        self.fc = nn.Linear(8,24)
                    
        # self.city_model_set = nn.ModuleList(CityModel(self.aqi_em, self.poi_em, self.wea_em,
        #                                          self.rnn_h, self.rnn_l, self.gnn_h, 
        #                                          self.hist_len, self.pred_len).to(self.device) for i in range(3))
        
        self.city_model_set = nn.ModuleList(CityModelv2(self.aqi_em, self.poi_em, self.wea_em,
                                                 self.rnn_h, self.rnn_l, self.gnn_h, 
                                                 self.hist_len, self.pred_len, self.fc).to(self.device) for i in range(self.valid_num))
    
    def forward(self, model_input):
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
        x_hist_tmp = x_hist
        c_x_hist, c_features = get_from_input(model_input[1])

        ### 测试
        # x_hist = x_hist.squeeze()
        # x_hist = x_hist.permute(0,2,1)
        # y_pred = self.fc(x_hist)
        # y_pred = y_pred.permute(0,2,1)[:,:,:,None]
        # sta_mark_set = np.arange(0, 152)
        ###

        cities_aqi = c_x_hist
        cities_conn = self.city_edge_index
        cities_sim = self.city_edge_attr

        city_u = self.global_model(cities_aqi, cities_conn, cities_sim,
                                self.city_num)

        city_outputs_set = []
        sta_mark_set = []
        for i, city_model in enumerate(self.city_model_set):
            valid_idx = self.valid_city_idx[i] 
            city_data = []
            sta_mark = self.afc_sparse[valid_idx]
            city_data.append(x_hist[:,:,sta_mark,:].permute(0,2,1,3))
            edge_index = torch.LongTensor(self.edge_index_set[i]).to(self.device)
            edge_attr = torch.Tensor(np.float32(self.edge_attr_set[i])).to(self.device)
            city_data.append(edge_index[None,:].repeat(x_hist.shape[0], 1, 1).permute(0,2,1))
            city_data.append(edge_attr[None, None, :].repeat(x_hist.shape[0], x_hist.shape[1], 1, 1))
            city_data.append(c_features[:, :self.hist_len, valid_idx, :])
            city_data.append(c_features[:, self.hist_len:, valid_idx, :])
            city_outputs = city_model(city_data, city_u[:, :, valid_idx], self.device)
            city_outputs_set.append(city_outputs)
            sta_mark_set = sta_mark_set + sta_mark

        city_outputs_set = torch.cat(city_outputs_set, dim=1)[:,:,:,None].permute(0,2,1,3)

        ### 测试
        # city_data = []
        # sta_mark = self.afc_sparse[0]
        # city_data.append(x_hist_tmp[:,:,:,:])
        # edge_index = torch.LongTensor(self.edge_index_set[0]).to(self.device)
        # edge_attr = torch.Tensor(np.float32(self.edge_attr_set[0])).to(self.device)
        # city_data.append(edge_index[None,:].repeat(x_hist_tmp.shape[0], 1, 1).permute(0,2,1))
        # city_data.append(edge_attr[None, None, :].repeat(x_hist_tmp.shape[0], x_hist_tmp.shape[1], 1, 1))
        # city_data.append(c_features[:, :self.hist_len, 0, :])
        # city_data.append(c_features[:, self.hist_len:, 0, :])
        # city_outputs = self.city_model_set[0](city_data, city_u[:, :, 0], self.device)
        # sta_mark_set = np.arange(0, 152)

        # city_outputs_set = city_outputs
        ###

        return city_outputs_set, sta_mark_set

        # return y_pred, sta_mark_set
    

'''
cities_aqi = c_x_hist
cities_conn = self.city_edge_index
cities_sim = self.city_edge_attr

city_u = self.global_model(cities_aqi, cities_conn, cities_sim,
                        self.city_num)

city_outputs_set = []
sta_mark_set = []
for idx, city_model in enumerate(self.city_model_set):
    city_data = []
    sta_mark = self.afc_sparse[idx]
    city_data.append(x_hist[:,:,sta_mark,:].permute(0,2,1,3))
    edge_index = torch.LongTensor(self.edge_index_set[idx]).to(self.device)
    edge_attr = torch.Tensor(np.float32(self.edge_attr_set[idx])).to(self.device)
    city_data.append(edge_index[None,:].repeat(x_hist.shape[0], 1, 1).permute(0,2,1))
    city_data.append(edge_attr[None, None, :].repeat(x_hist.shape[0], x_hist.shape[1], 1, 1))
    city_data.append(c_features[:, :self.hist_len, idx, :])
    city_data.append(c_features[:, self.hist_len:, idx, :])
    city_outputs = city_model(city_data, city_u[:, :, idx], self.device)
    city_outputs_set.append(city_outputs)
    sta_mark_set = sta_mark_set + sta_mark

city_outputs_set = torch.cat(city_outputs_set, dim=1)[:,:,:,None].permute(0,2,1,3)
'''