import os
import sys
proj_dir = os.getcwd()
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham
import folium
import argparse


class ProEdge():
    def __init__(self, nodes_fp, adj_fp, save_fg=False):
        self.use_altitude = False
        self.nodes = np.load(nodes_fp, allow_pickle=True) # DataFrame
        self.adj = np.load(adj_fp)
        self.edge_index, self.edge_attr = self._gen_edges()
        if self.use_altitude:
            self._update_edges() # filtered by altitude<alti_thres ---> get valid edges
        if save_fg:
            np.save('city_edge_index.npy', self.edge_index)
            np.save('city_edge_attr.npy', self.edge_attr)

    def _gen_edges(self):
        adj = self.adj - np.identity(self.adj.shape[1])
        adj_abs = adj
        adj_abs[adj_abs > 0] = 1
        assert len(adj > 0) == len(adj_abs == 1)
        edge_index, _ = dense_to_sparse(torch.tensor(adj_abs)) # edge_index: (2, valid_edge_num)
        edge_index = edge_index.numpy()

        direc_arr = []
        dist_km_arr =[]
        for i in range(edge_index.shape[1]): # traverse valid_edge_num
            src, dest = edge_index[0, i], edge_index[1, i]
            src_lat, src_lon = self.nodes.iloc[src, :]['lat'], self.nodes.iloc[src, :]['lon']
            dest_lat, dest_lon = self.nodes.iloc[dest, :]['lat'], self.nodes.iloc[dest, :]['lon']
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-info-fp", type=str, default="/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_152/station_hb152.pkl")
    parser.add_argument("--adj-info-fp", type=str, default="/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_152/station_hb152_adj.npy")
    args = parser.parse_args()
    '''
    station:
    parser.add_argument("--node-info-fp", type=str, default="/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_152/station_hb152.pkl")
    parser.add_argument("--adj-info-fp", type=str, default="/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_152/station_hb152_adj.npy")

    city:
    parser.add_argument("--node-info-fp", type=str, default="/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_city_41_station_list.pkl")
    parser.add_argument("--adj-info-fp", type=str, default="/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_152/adj_c.npy")
    '''
    pro_edge = ProEdge(args.node_info_fp, args.adj_info_fp, save_fg=False)