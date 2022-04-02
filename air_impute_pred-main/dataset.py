'''
city-level data:
    mean;
    Afc;
    adj DCRNN
    city-level supervised info: designed 24 hours may not feasible
'''
import os
import sys
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

import numpy as np
import pandas as pd
import hickle as hkl
from torch.utils import data
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham
import folium
import torch


class HazeDataset(data.Dataset):

    def __init__(self, rawdata, data_type):
        self.rawdata = rawdata

        self.air = self.rawdata.dataset[data_type]['air']
        self.met = self.rawdata.dataset[data_type]['met']
        self.time = self.rawdata.dataset[data_type]['time']
        self.t = self.rawdata.dataset[data_type]['t']

        self.enc_len = self.rawdata.conf.param['train']['enc_len']
        self.dec_len = self.rawdata.conf.param['train']['dec_len']
        self.seq_len = self.enc_len + self.dec_len

        self.mean = self.rawdata.dataset['mean']
        self.std = self.rawdata.dataset['std']

        self.air = (self.air - self.mean['air']) / self.std['air']
        self.met = (self.met - self.mean['met']) / self.std['met']
        self.time = (self.time - self.mean['time']) / self.std['time']

        self.t = self._to_batch(self.t)
        self.air = self._to_batch(self.air)
        self.met = self._to_batch(self.met)
        self.time = self._to_batch(self.time)

    def _to_batch(self, arr):
        arr_batch = []
        t_offsets = np.arange(0, self.seq_len)
        for i in range(len(arr) - self.seq_len):
            idxs = t_offsets + i
            arr_batch.append(arr[idxs])
        arr_batch = np.stack(arr_batch).astype(np.float32)
        return arr_batch

    def __len__(self):
        return len(self.air)

    def __getitem__(self, index):
        return self.air[index, :self.enc_len], \
               self.met[index, :self.enc_len], \
               self.time[index, :self.enc_len], \
               self.t[index, :self.enc_len], \
               self.air[index, self.enc_len:], \
               self.met[index, self.enc_len:], \
               self.time[index, self.enc_len:], \
               self.t[index, self.enc_len:]

class HazeCoarseDataset(data.Dataset):
    
    def __init__(self, rawdata, data_type):
        self.rawdata = rawdata
        self.enc_len = self.rawdata.conf.param['train']['enc_len']
        self.dec_len = self.rawdata.conf.param['train']['dec_len']
        self.seq_len = self.enc_len + self.dec_len

        self.air = self.rawdata.dataset[data_type]['air']
        self.met = self.rawdata.dataset[data_type]['met']
        self.time = self.rawdata.dataset[data_type]['time']
        self.t = self.rawdata.dataset[data_type]['t']

        self.city_air = self.rawdata.city_dataset[data_type]['air']
        self.city_met = self.rawdata.city_dataset[data_type]['met']
        self.city_time = self.rawdata.city_dataset[data_type]['time']
        self.city_t = self.rawdata.city_dataset[data_type]['t']

        self.mean = self.rawdata.dataset['mean']
        self.std = self.rawdata.dataset['std']
        self.city_mean = self.rawdata.city_dataset['mean']
        self.city_std = self.rawdata.city_dataset['std']

        self.air = (self.air - self.mean['air']) / self.std['air']
        self.met = (self.met - self.mean['met']) / self.std['met']
        self.time = (self.time - self.mean['time']) / self.std['time']

        self.city_air = (self.city_air - self.city_mean['air']) / self.city_std['air']
        self.city_met = (self.city_met - self.city_mean['met']) / self.city_std['met']
        self.city_time = (self.city_time - self.city_mean['time']) / self.city_std['time']

        self.t = self._to_batch(self.t)
        self.air = self._to_batch(self.air)
        self.met = self._to_batch(self.met)
        self.time = self._to_batch(self.time)

        self.city_t = self._to_batch(self.city_t)
        self.city_air = self._to_batch(self.city_air)
        self.city_met = self._to_batch(self.city_met)
        self.city_time = self._to_batch(self.city_time)

    def _to_batch(self, arr):
        arr_batch = []
        t_offsets = np.arange(0, self.seq_len)
        for i in range(len(arr) - self.seq_len):
            idxs = t_offsets + i
            arr_batch.append(arr[idxs])
        arr_batch = np.stack(arr_batch).astype(np.float32)
        return arr_batch

    def __len__(self):
        return len(self.air)

    def __getitem__(self, index):
        all_data = []
        sta_data = self.air[index, :self.enc_len], \
                self.met[index, :self.enc_len], \
                self.time[index, :self.enc_len], \
                self.t[index, :self.enc_len], \
                self.air[index, self.enc_len:], \
                self.met[index, self.enc_len:], \
                self.time[index, self.enc_len:], \
                self.t[index, self.enc_len:]
        city_data = self.city_air[index, :self.enc_len], \
                self.city_met[index, :self.enc_len], \
                self.city_time[index, :self.enc_len], \
                self.city_t[index, :self.enc_len], \
                self.city_air[index, self.enc_len:], \
                self.city_met[index, self.enc_len:], \
                self.city_time[index, self.enc_len:], \
                self.city_t[index, self.enc_len:]
        
        all_data.append(sta_data)
        all_data.append(city_data)
        return all_data

class RawData(object):

    def __init__(self, conf):

        self.conf = conf
        self.t_offset = pd.to_datetime('2014-01-01 00:00:00', format='%Y-%m-%d %H:%M:%S').tz_localize('Asia/Shanghai')

        self.air_var = self.conf.param['data']['air_var']
        self.met_var = self.conf.param['data']['met_var']
        self.t_var = self.conf.param['data']['t_var']

        self.var = self.air_var.copy()
        if self.conf.param['data']['use_met']:
            self.var += self.met_var
        if self.conf.param['data']['use_time']:
            self.var += self.t_var

        self.dataset_fp = os.path.join(self.conf.root_dir, 'hb_152/air_dataset_knn_hb.pkl')
        self.data = pd.read_pickle(self.dataset_fp)

        # {'O3', 'PM10_knn', 'O3_knn', 'NO2_knn', 'sp', 't2m', 'rh2m', 'tp', 'PM2.5', 'blh', 'NO2', 'msdwswrf', 'u100',
        #  'dir100', 'v100', 'spd100', 'd2m', 'PM10', 'PM2.5_knn'}

        self.node_info_fp = os.path.join(self.conf.root_dir, 'hb_152/station_hb152.pkl')
        self.city_info_fp = os.path.join(self.conf.root_dir, 'hb_city_41_station_list.pkl')
        self.adj_sta_fp = os.path.join(self.conf.root_dir, 'hb_152/station_hb152_adj.npy')
        self.adj_city_fp = os.path.join(self.conf.root_dir, 'hb_152/adj_city.npy')
        self.afc_s2c_fp = os.path.join(self.conf.root_dir, 'hb_152/afc_s2c.npy')
        self.sta_edge_index_fp = os.path.join(self.conf.root_dir, 'hb_152/sta_edge_index.npy')
        self.sta_edge_attr_fp = os.path.join(self.conf.root_dir, 'hb_152/sta_edge_attr.npy')
        self.city_edge_index_fp = os.path.join(self.conf.root_dir, 'hb_152/city_edge_index.npy')
        self.city_edge_attr_fp = os.path.join(self.conf.root_dir, 'hb_152/city_edge_attr.npy')

        self.node_info = pd.read_pickle(self.node_info_fp)
        self.city_info = pd.read_pickle(self.city_info_fp)
        self.adj_sta = np.load(self.adj_sta_fp)
        self.adj_city = np.load(self.adj_city_fp)
        self.afc_s2c = np.load(self.afc_s2c_fp)

        def afc2sparse(afc):
            afc_sparse = []
            for j in range(len(afc[0, :])):
                col = []
                for i in range(len(afc)):
                    if afc[i, j] == 1:
                        col.append(i)
                afc_sparse.append(col)
            return afc_sparse

        self.afc_sparse = afc2sparse(self.afc_s2c)

        # sta_edge用在HighAir中需要筛选, 同一city内才相连
        if conf.param['experiments']['model_use'] == 'HighAir':
            def get_adj(lon, lat):
                lon = np.array(lon)
                lat = np.array(lat)
                normalized_k = 0.2
                coordinates_array = np.stack((lat, lon), axis=1)
                num_node = lon.shape[0]
                dist_mx = np.zeros((num_node, num_node))
                for i in range(num_node):
                    for j in range(num_node):
                        coord_i = coordinates_array[i,]
                        coord_j = coordinates_array[j,]
                        dist_mx[i, j] = geodesic(coord_i, coord_j).kilometers

                # Calculates the standard deviation as theta.
                distances = dist_mx[~np.isinf(dist_mx)].flatten()
                std = distances.std()
                adj_mx = np.exp(-np.square(dist_mx / std))
                # Make the adjacent matrix symmetric by taking the max.
                # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

                # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
                adj_mx[adj_mx < normalized_k] = 0
                return adj_mx

            def get_edge(nodes, adj):
                adj = adj - np.identity(adj.shape[1])
                adj_abs = adj
                adj_abs[adj_abs > 0] = 1
                assert len(adj > 0) == len(adj_abs == 1)
                edge_index, _ = dense_to_sparse(torch.tensor(adj_abs)) # edge_index: (2, valid_edge_num)
                edge_index = edge_index.numpy()

                direc_arr = []
                dist_km_arr =[]
                for i in range(edge_index.shape[1]): # traverse valid_edge_num
                    src, dest = edge_index[0, i], edge_index[1, i]
                    src_lat, src_lon = nodes.iloc[src, :]['lat'], nodes.iloc[src, :]['lon']
                    dest_lat, dest_lon = nodes.iloc[dest, :]['lat'], nodes.iloc[dest, :]['lon']
                    src_loc = (src_lat, src_lon)
                    dest_loc = (dest_lat, dest_lon)
                    dist_km = geodesic(src_loc, dest_loc).kilometers

                    def posi_nega_fg(value):
                        if abs(value) == 0:
                            print("same lat/lon")
                            return 0
                        else:
                            return value/abs(value)

                    v = posi_nega_fg(src_lat - dest_lat) * geodesic((src_lat, dest_lon), (dest_lat, dest_lon)).meters # keep the longitude consistent
                    u = posi_nega_fg(src_lon - dest_lon) * geodesic((src_lat, src_lon), (src_lat, dest_lon)).meters # Keep the latitude consistent

                    u = u * units.meter / units.second
                    v = v * units.meter / units.second
                    direc = mpcalc.wind_direction(u, v)._magnitude # direction form source node to destination node
                    direc_arr.append(direc)
                    dist_km_arr.append(dist_km)

                direc_arr = np.stack(direc_arr) 
                dist_km_arr = np.stack(dist_km_arr)
                attr = np.stack([dist_km_arr, direc_arr], axis=-1) # (valid_edge_num, 2)

                return edge_index, attr

            self.edge_index_set = []
            self.edge_attr_set = []
            self.valid_city_idx = []
            for i in range(self.afc_s2c.shape[1]):
                sta_mark = self.afc_sparse[i]
                if len(sta_mark) <= 1:
                    # self.edge_index_set.append(-1)
                    # self.edge_attr_set.append(-1)
                    continue
                adj_mx = get_adj(self.node_info.iloc[sta_mark,:]['lon'], self.node_info.iloc[sta_mark,:]['lat'])
                if (adj_mx == np.identity(adj_mx.shape[1])).all():
                    # self.edge_index_set.append(-1)
                    # self.edge_attr_set.append(-1)
                    continue
                index, attr = get_edge(self.node_info.iloc[sta_mark,:], adj_mx)
                self.edge_index_set.append(index)
                self.edge_attr_set.append(attr)
                self.valid_city_idx.append(i)

        # afc_s2c / D
        self.D = np.sum(self.afc_s2c, axis=0)
        self.afc_s2c = self.afc_s2c / self.D

        self.sta_edge_index = np.load(self.sta_edge_index_fp)
        self.sta_edge_attr = np.load(self.sta_edge_attr_fp)
        self.city_edge_index = np.load(self.city_edge_index_fp)
        self.city_edge_attr = np.load(self.city_edge_attr_fp)

        # normlize edge_attr
        sta_edge_attr_mean = self.sta_edge_attr.mean()
        sta_edge_attr_std = self.sta_edge_attr.std()
        self.sta_edge_attr = (self.sta_edge_attr - sta_edge_attr_mean) / sta_edge_attr_std

        city_edge_attr_mean = self.city_edge_attr.mean()
        city_edge_attr_std = self.city_edge_attr.std()
        self.city_edge_attr = (self.city_edge_attr - city_edge_attr_mean) / city_edge_attr_std

        ### resample to 3 hours
        def using_Grouper(df):
            level_values = df.index.get_level_values
            return (df.groupby([pd.Grouper(freq='3H', level=0)] + [level_values(i) for i in [1]]).mean())
        if self.conf.param['train']['time_interval'] != 1:
            self.data = using_Grouper(self.data)
        else:
            pass

        # print(self.data[0:20])
        # print(self.adj[0:20])
        # print(self.node_info[0:20])
        
        self.city_info_array = self.city_info.loc[:,['lat', 'lon']].to_numpy()
        city_info_mean = self.city_info_array.mean(axis=0)
        city_info_std = self.city_info_array.std(axis=0)
        self.city_info_norm_array = (self.city_info_array - city_info_mean) / city_info_std

        self.node = self.data.columns.tolist()
        self.dataset, self.city_dataset = self._load_rawdata()

    def _softmax(self, x):           
        max = np.max(x, axis=1, keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x, axis=1, keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x

    def t_str(self, t):
        return pd.to_datetime(t, format='%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H')
    
    def _info_s2c(self, ):

        return

    def _load_rawdata(self):
        data = {}
        meanstd = {} 
        city_data = {}
        city_meanstd = {}

        for typ in ['train', 'val', 'test']:
            preiod_select = self.conf.param['train']['preiod_select']
            t_typ = self.conf.param['data']['period'][preiod_select][typ]
            t_start = t_typ[0]
            t_end = t_typ[1]
            t_h = pd.date_range(start=t_start, end=t_end, freq='H', tz='Asia/Shanghai')

            dataset_air = []
            for v in self.air_var:
                v = '%s_knn' % v
                data_v = self.data.xs(v, level=1) # 筛选指标
                data_v_t = data_v[t_start: t_end].values # 筛选时间段
                dataset_air.append(data_v_t)
            dataset_air = np.stack(dataset_air, -1)

            dataset_met = []
            for v in self.met_var:
                data_v = self.data.xs(v, level=1)
                data_v_t = data_v[t_start: t_end].values
                dataset_met.append(data_v_t)
            dataset_met = np.stack(dataset_met, -1)

            dataset_time = []
            for v in self.t_var:

                if v == 'dow':
                    dow = t_h.dayofweek
                    data_v_t = np.stack([dow] * len(self.node), axis=1)
                if v == 'hour':
                    h = t_h.hour
                    data_v_t = np.stack([h] * len(self.node), axis=1)

                dataset_time.append(data_v_t)
            dataset_time = np.stack(dataset_time, -1)

            dataset_t = (t_h - self.t_offset) / pd.Timedelta('1 hour')

            ### city-level: by mean
            afc_s2c_sm = self._softmax(self.afc_s2c.T)
            city_dataset_air = np.matmul(afc_s2c_sm, dataset_air)
            city_dataset_met = np.matmul(afc_s2c_sm, dataset_met)
            city_dataset_time = np.matmul(afc_s2c_sm, dataset_time)
            city_dataset_t = dataset_t

            def get_data(typ, dataset_air, dataset_met, dataset_time, dataset_t, data, meanstd):
                if typ == 'train':
                    air_mean = np.nanmean(dataset_air, axis=(0, 1))
                    air_std = np.nanstd(dataset_air, axis=(0, 1))

                    met_mean = np.nanmean(dataset_met, axis=(0, 1))
                    met_std = np.nanstd(dataset_met, axis=(0, 1))

                    t_mean = np.nanmean(dataset_time, axis=(0, 1))
                    t_std = np.nanstd(dataset_time, axis=(0, 1))

                    meanstd.update({
                        'mean': {
                            'air': air_mean,
                            'met': met_mean,
                            'time': t_mean,
                        },
                        'std': {
                            'air': air_std,
                            'met': met_std,
                            'time': t_std,
                        },
                    })

                data.update(meanstd)
                data.update(
                    {typ: {'air': dataset_air, 'met': dataset_met, 'time': dataset_time, 't': dataset_t}}
                )
                return data, meanstd
            
            data, meanstd= get_data(typ, dataset_air, dataset_met, dataset_time, dataset_t, data, meanstd)
            city_data, city_meanstd = get_data(typ, city_dataset_air, city_dataset_met, city_dataset_time, city_dataset_t, city_data, city_meanstd)

        return data, city_data


if __name__ == '__main__':

    from util import Config

    conf = Config(os.path.join(PROJ_DIR, 'config/conf.yaml'))

    rawdata = RawData(conf)
    trainset = HazeDataset(rawdata, 'train')
