import numpy as np
import pandas as pd
import numpy_indexed as npi

df_c2s = pd.read_pickle('/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_city_41_station_list.pkl')
df_node_info = pd.read_pickle('/home/tokaka22/Multi_scale_GNN/data/knn-data/hb_152/station_hb152.pkl')

station2index = df_node_info.iloc[:,0].to_numpy()

sta_idx = []
for i in range(len(df_c2s)):
    sta_idx.append(npi.indices(station2index, df_c2s.iloc[i,-1]))

df_c2s['sta_idx'] = sta_idx

# df_node_info.to_csv('df_node_info.csv', encoding="utf_8_sig")
# df_c2s.to_csv('df_c2s_idx.csv', encoding="utf_8_sig")

afc = np.zeros((len(df_c2s), len(station2index)))
    
for i in range(len(df_c2s)):
    city_idx = i
    sta_idx_set = df_c2s.iloc[i, -1]
    for sta_idx in sta_idx_set:
        afc[i, sta_idx] = 1

assert afc.sum() == len(station2index)
np.save("afc.npy", afc.T)
# np.savetxt("afc.csv", afc, delimiter=',')

print("ok")