import os
import sys
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

import yaml
import numpy as np
import pandas as pd

import getpass
import torch
import argparse
import json


class Config():

    def __init__(self, args):
        conf_fp = os.path.join(PROJ_DIR, args.config)

        print('Parse config: %s' % conf_fp)

        with open(conf_fp) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        ### 读取yaml文件(一些基本不变的值)
        self.param = config['hyperparameter']
        self.info = config['info']

        self.root_dir = config['filepath']['root_dir']
        self.data_dir = os.path.join(PROJ_DIR, 'data')

        ### argparse进一步修改某些值
        self.param['gpu_id'] = args.gpu_id
        self.info['group_name'] = args.group_name
        self.info['wandb_proj_name'] = args.project_name
        self.param['train']['model_use'] = args.model_use
        self.param['train']['preiod_select'] = args.preiod_select
        self.param['train']['seed'] = args.seed

        self.param['train']['enc_len'] = args.enc_len
        self.param['train']['dec_len'] = args.dec_len

        self.param['train']['epoch'] = args.epoch
        self.param['train']['early_stop'] = args.early_stop
        self.param['train']['batch_size'] = args.batch_size
        self.param['train']['wd'] = args.wd
        self.param['train']['lr'] = args.lr
        self.param['train']['hid_dim'] = args.hid_dim
        self.param['train']['mt_hid_dim'] = args.mt_hid_dim
        self.param['train']['gnn_out_dim'] = args.gnn_out_dim
        self.param['train']['c_weight'] = args.c_weight
        self.param['train']['k_hop'] = args.k_hop
        self.param['train']['time_interval'] = args.time_interval
        self.param['train']['cheb_norm'] = args.cheb_norm
        self.param['train']['interval_set'] = args.interval_set

        if type(self.param['gpu_id']) is int:
            self.gpu_id = self.param['gpu_id']
        # 自动选择空闲memory最大的显卡
        else: 
            use_cuda = torch.cuda.is_available()

            if use_cuda:
                # 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
                os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
                memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
                print(f"memory_gpu: {memory_gpu}")
                gpu_ids = str(np.argmax(memory_gpu)) 
                print(f"CUDA_VISIBLE_DEVICES: {gpu_ids}")
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
                os.system('rm tmp') # 删除临时生成的 tmp 文件

            self.gpu_id = int(gpu_ids)  # os.environ["CUDA_VISIBLE_DEVICES"]

        print(f"self.gpu_id: {self.gpu_id}")

        ### 使用city数据
        if self.param['train']['model_use'] in ['CityGNN', 'MTCityGNN', 'MTClusterGNN', 'MTCityGNNv2', 'City_GC_GRU', 'MTCity_GC_GRU', 'City_GC_GRUv2', 'MTCity_GC_GRUv2', 'City_GC_GRUv3', 'HighAir', 'HighAirv2']:
            self.param['city_data_use_flag'] = True
        else:
            self.param['city_data_use_flag'] = False
        
        ### MT网络hid_dim根据interval_set自动增大
        if self.param['train']['model_use'] in ['MT_GC_GRU', 'MTCity_GC_GRU']:
            self.param['train']['hid_dim'] = self.param['train']['mt_hid_dim'] * len(self.param['train']['interval_set'])
            self.param['train']['mt_hid_dim'] = self.param['train']['hid_dim']

        ### MT网络hid_dimz直接设置为mt_hid_dim
        # if self.param['train']['model_use'] in ['MT_GC_GRU', 'MTCity_GC_GRU']:
        #     self.param['train']['hid_dim'] = self.param['train']['mt_hid_dim']

        self.t_now = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self._construct_dirs(args.run_idx)

    def _construct_dirs(self, run_idx):
        self.log_dir = os.path.join(self.root_dir, 'results', self.t_now, '%s_%s' % (self.param['train']['model_use'], str(run_idx)))
        os.makedirs(self.log_dir, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Air Fusion & Imputation')
    parser.add_argument('--config', type=str, default='config/conf.yaml', help='config file')
    parser.add_argument('--enc-len', type=int, default=8, help='')
    parser.add_argument('--dec-len', type=int, default=32, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--gpu-id', type=int, default=0, help='')
    parser.add_argument('--project-name', type=str, default='air_impute_p5_trash', help='')
    parser.add_argument('--model-use', type=str, default='GC_GRU', help='')
    parser.add_argument('--k-hop', type=int, default=1, help='')
    parser.add_argument('--run-idx', type=int, default=0, help='')
    parser.add_argument('--seed', type=int, default=6, help='')
    args = parser.parse_args()

    ### yaml作为default; argparse进一步修改某些值
    conf = Config(args)
