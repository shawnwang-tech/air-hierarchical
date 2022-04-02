import os
import sys
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

import argparse

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from models.gru import GRU
from models.dcrnn import DCRNN
from models.stgcn import STGCN
from models.graphwavenet import GWNet
from models.MSTGCN import MSTGCN
from models.MTClusterGNN import MTClusterGNN
from models.MTCityGNN import MTCityGNN
from models.GC_GRU import GC_GRU
from models.MT_GC_GRU import MT_GC_GRU 
from models.MTCity_GC_GRU import MTCity_GC_GRU
from models.MTCity_GC_GRUv2 import MTCity_GC_GRUv2 
from models.City_GC_GRU import City_GC_GRU 
from models.City_GC_GRUv2 import City_GC_GRUv2 
from models.City_GC_GRUv3 import City_GC_GRUv3
from models.MTCityGNNv2 import MTCityGNNv2
from models.CityGNN import CityGNN 

from models.HighAir.HighAirv3v1 import HighAirv2

from models.ASTGCN import ASTGCN
from models.PM25GNN import PM25GNN
from models.MTMSGNN import MTMSGNN 
from models.MSGNN import MSGNN
from models.HighAir.HighAir import HighAir

from dataset import RawData, HazeDataset, HazeCoarseDataset
from util import Config
from metrics import LightningMetric


class LightningData(LightningDataModule):
    def __init__(self, trainset, valset, testset):
        super().__init__()
        self.batch_size = conf.param['train']['batch_size']
        self.trainset = trainset
        self.valset = valset
        self.testset = testset

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=conf.param['train']['num_workers'], pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=conf.param['train']['num_workers'], pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=conf.param['train']['num_workers'], pin_memory=True, drop_last=True)


class LightningModel(LightningModule):

    def __init__(self, log_dir, wandb_logger, param, gpu_id=0):
        super().__init__()
        # store all the provided arguments under the self.hparams attribute.
        # loggers that support it will automatically log the contents of self.hparams.
        self.param = param
        self.save_hyperparameters() # 保存hyparam在wandb的log最后面 

        self.gpu_id = gpu_id
        self.log_dir = log_dir
        self.cuda_device = torch.device('cuda:%s' % self.gpu_id)
        self.var_use = rawdata.var

        ########## Data ###########
        self.use_time = rawdata.conf.param['data']['use_time']
        self.use_met = rawdata.conf.param['data']['use_met']

        self.in_dim = len(trainset.rawdata.air_var)
        if self.use_met:
            self.in_dim += len(trainset.rawdata.met_var)
        if self.use_time:
            self.in_dim += len(trainset.rawdata.t_var)
        self.out_dim = len(trainset.rawdata.air_var)
        ########## Data ###########

        ########## Graph ##########

        if conf.param['train']['precision'] == 16:
            self.graph = torch.HalfTensor(rawdata.adj_sta).to(self.cuda_device)
        else:
            self.graph = torch.Tensor(rawdata.adj_sta).to(self.cuda_device)

        ########## Graph ##########

        ########## Models ##########

        if conf.param['train']['model_use'] == 'GRU':

            self.model = GRU(
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len
            )
        
        elif conf.param['train']['model_use'] == 'GC_GRU':

            self.model = GC_GRU(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['gnn_out_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                conf.param['train']['k_hop'],
                conf.param['train']['interval_set']
            )
        
        elif conf.param['train']['model_use'] == 'MT_GC_GRU':

            self.model = MT_GC_GRU(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['gnn_out_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
                conf.param['train']['interval_set']
            )

        elif conf.param['train']['model_use'] == 'City_GC_GRU':

            self.model = City_GC_GRU(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['gnn_out_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                rawdata.adj_city,
                rawdata.afc_s2c,
                conf.param['train']['k_hop'],
                conf.param['train']['interval_set']
            )
        
        elif conf.param['train']['model_use'] == 'City_GC_GRUv2':

            self.model = City_GC_GRUv2(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                rawdata.adj_city,
                rawdata.afc_s2c,
                conf.param['train']['k_hop']
            )
        
        elif conf.param['train']['model_use'] == 'City_GC_GRUv3':

            self.model = City_GC_GRUv3(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                rawdata.adj_city,
                rawdata.afc_s2c,
                conf.param['train']['k_hop']
            )
        
        elif conf.param['train']['model_use'] == 'MTCity_GC_GRU': 

            self.model = MTCity_GC_GRU(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['gnn_out_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                rawdata.adj_city,
                rawdata.afc_s2c,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
                conf.param['train']['interval_set']
            )

        elif conf.param['train']['model_use'] == 'MTCity_GC_GRUv2': 

            self.model = MTCity_GC_GRUv2(
                self.gpu_id,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['batch_size'],
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.adj_sta,
                rawdata.adj_city,
                rawdata.afc_s2c,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop']
            )

        elif conf.param['train']['model_use'] == 'DCRNN':

            self.model = DCRNN(
                self.gpu_id,
                conf.param['train']['batch_size'],
                self.in_dim,
                self.out_dim,
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                self.graph,
                len(self.graph)
            )

        elif conf.param['train']['model_use'] == 'STGCN':

            self.model = STGCN(
                self.gpu_id,
                conf.param['train']['batch_size'],
                self.in_dim,
                self.out_dim,
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                self.graph,
                len(self.graph)
            )

        elif conf.param['train']['model_use'] == 'GWNet':

            self.model = GWNet(
                self.gpu_id,
                conf.param['train']['batch_size'],
                self.in_dim,
                self.out_dim,
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                self.graph,
                len(self.graph)
            )

        elif conf.param['train']['model_use'] == 'MSTGCN':

            self.model = MSTGCN(
                self.gpu_id,
                conf.param['train']['batch_size'],
                self.in_dim,
                self.out_dim,
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                self.graph,
                len(self.graph)
            )

        elif conf.param['train']['model_use'] == 'ASTGCN':

            self.model = ASTGCN(
                self.gpu_id,
                conf.param['train']['batch_size'],
                self.in_dim,
                self.out_dim,
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                self.graph,
                len(self.graph)
            )
        
        elif conf.param['train']['model_use'] == 'PM25GNN':
    
            self.model = PM25GNN(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['gnn_out_dim'],
                len(self.graph),
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
            )

        elif conf.param['train']['model_use'] == 'MTMSGNN':
    
            self.model = MTMSGNN(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                len(self.graph),
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop']
            )

        elif conf.param['train']['model_use'] == 'MSGNN':
        
            self.model = MSGNN(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.afc_s2c,
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop']
            )

        elif conf.param['train']['model_use'] == 'MTClusterGNN':
            
            self.model = MTClusterGNN(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.afc_s2c,
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.city_edge_index,
                rawdata.city_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                rawdata.city_dataset['mean']['met'][1:3],
                rawdata.city_dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
                rawdata.city_info_norm_array,
                rawdata.city_info_array,
                conf.param['model']['MTClusterGNN']['num_cluster']
            )

        elif conf.param['train']['model_use'] == 'MTCityGNN':
            
            self.model = MTCityGNN(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.afc_s2c,
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.city_edge_index,
                rawdata.city_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                rawdata.city_dataset['mean']['met'][1:3],
                rawdata.city_dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
            )   
        
        elif conf.param['train']['model_use'] == 'MTCityGNNv2':
            
            self.model = MTCityGNNv2(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.afc_s2c,
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.city_edge_index,
                rawdata.city_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                rawdata.city_dataset['mean']['met'][1:3],
                rawdata.city_dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
            )   
        
        elif conf.param['train']['model_use'] == 'CityGNN':
            
            self.model = CityGNN(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.afc_s2c,
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.city_edge_index,
                rawdata.city_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                rawdata.city_dataset['mean']['met'][1:3],
                rawdata.city_dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
            )   
        
        elif conf.param['train']['model_use'] == 'HighAirv2':
            
            self.model = HighAirv2(
                self.gpu_id,
                trainset.enc_len,
                trainset.dec_len,
                self.in_dim,
                self.out_dim,
                conf.param['train']['hid_dim'],
                conf.param['train']['gnn_out_dim'],
                len(rawdata.adj_sta),
                len(rawdata.adj_city),
                rawdata.afc_s2c,
                conf.param['train']['batch_size'],
                rawdata.sta_edge_index,
                rawdata.sta_edge_attr,
                rawdata.city_edge_index,
                rawdata.city_edge_attr,
                rawdata.dataset['mean']['met'][1:3],
                rawdata.dataset['std']['met'][1:3],
                rawdata.city_dataset['mean']['met'][1:3],
                rawdata.city_dataset['std']['met'][1:3],
                self.use_met,
                self.use_time,
                conf.param['train']['interval_set'],
                conf.param['train']['k_hop'],
            )   
        
        elif conf.param['train']['model_use'] == 'HighAir':
            
            self.model = HighAir(
                self.gpu_id,
                conf.param['model']['HighAir']['aqi_em'],
                conf.param['model']['HighAir']['poi_em'],
                conf.param['model']['HighAir']['wea_em'],
                conf.param['model']['HighAir']['rnn_h'],
                conf.param['model']['HighAir']['rnn_l'],
                conf.param['model']['HighAir']['gnn_h'],
                len(rawdata.adj_city),
                len(rawdata.adj_sta),
                rawdata.city_edge_index,
                rawdata.city_edge_attr,
                rawdata.edge_index_set,
                rawdata.edge_attr_set,
                self.use_met,
                self.use_time,
                trainset.enc_len,
                trainset.dec_len,
                rawdata.afc_sparse,
                rawdata.valid_city_idx
            )  

        else:
            raise NotImplementedError

        if conf.param['train']['orth_flag'] == False: # orthogonal_初始化gru
            for p in self.model.parameters(): # model所有参数初始化
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        
        wandb_logger.watch(self.model, log="all", log_freq=50)
        ########## Models ##########

        self.criterion = nn.MSELoss()

        self.metric_lightning = LightningMetric(self.log_dir, rawdata.dataset['mean']['air'], rawdata.dataset['std']['air'])


    def forward(self, X):
        return self.model(X)

    def _run_model(self, batch):
        enc_air, enc_met, enc_time, enc_t, dec_air, dec_met, dec_time, dec_t = batch

        enc = [enc_air]
        dec = []
        if self.use_met:
            enc += [enc_met]
            dec += [dec_met]
        if self.use_time:
            enc += [enc_time]
            dec += [dec_time]

        dec_air_hat = self([enc, dec])

        loss = self.criterion(dec_air_hat, dec_air)

        return dec_air_hat, dec_air, loss, dec_t

    def _c_run_model(self, batch, typ='val_test'):
        enc_air, enc_met, enc_time, enc_t, dec_air, dec_met, dec_time, dec_t = batch[0]
        c_enc_air, c_enc_met, c_enc_time, c_enc_t, c_dec_air, c_dec_met, c_dec_time, c_dec_t = batch[1]

        def get_enc_dec(enc_air, enc_met, dec_met, enc_time, dec_time):
            enc = [enc_air]
            dec = []
            if self.use_met:
                enc += [enc_met]
                dec += [dec_met]
            if self.use_time:
                enc += [enc_time]
                dec += [dec_time]
            return enc, dec

        enc, dec = get_enc_dec(enc_air, enc_met, dec_met, enc_time, dec_time)
        c_enc, c_dec = get_enc_dec(c_enc_air, c_enc_met, c_dec_met, c_enc_time, c_dec_time)

        if conf.param['train']['model_use'] == 'HighAir':
            dec_air_hat, sta_mark = self([[enc, dec], [c_enc, c_dec]])
        else:
            dec_air_hat, c_dec_air_hat = self([[enc, dec], [c_enc, c_dec]])
        
        if typ == 'train':
            if conf.param['train']['model_use'] == 'HighAir':
                dec_air = dec_air[:, :, sta_mark]
                loss = self.criterion(dec_air_hat, dec_air)
                return dec_air_hat, dec_air, loss, dec_t
            elif conf.param['train']['model_use'] == 'HighAirv2':
                loss_s = self.criterion(dec_air_hat, dec_air)
                loss = loss_s
                return dec_air_hat, dec_air, loss, dec_t
            else:
                loss_s = self.criterion(dec_air_hat, dec_air)
                loss_c = conf.param['train']['c_weight'] * self.criterion(c_dec_air_hat, c_dec_air)
                loss = loss_s + loss_c
                # loss = loss_s
                return dec_air_hat, dec_air, loss, dec_t, loss_s, loss_c
        else:
            if conf.param['train']['model_use'] == 'HighAir':
                dec_air = dec_air[:, :, sta_mark]
                loss = self.criterion(dec_air_hat, dec_air)
                return dec_air_hat, dec_air, loss, dec_t
            else:
                loss = self.criterion(dec_air_hat, dec_air)
                return dec_air_hat, dec_air, loss, dec_t

    def training_step(self, batch, batch_idx):
        if conf.param['city_data_use_flag'] == False:
            out, label, loss, t = self._run_model(batch)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            if conf.param['train']['model_use'] in ['HighAir', 'HighAirv2']:
                out, label, loss, t = self._c_run_model(batch, 'train')
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            else:
                out, label, loss, t, loss_s, loss_c = self._c_run_model(batch, 'train')
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('train_loss_s', loss_s, on_step=True, on_epoch=True, logger=True)
                self.log('train_loss_c', loss_c, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if conf.param['city_data_use_flag'] == False:
            out, label, loss, t = self._run_model(batch)
        else:
            out, label, loss, t = self._c_run_model(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        if conf.param['city_data_use_flag'] == False:
            out, label, loss, t = self._run_model(batch)
        else:
            out, label, loss, t = self._c_run_model(batch)
        self.metric_lightning.update(out, label, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def training_epoch_end(self, outputs):
        pass
        return

    def test_epoch_end(self, outputs):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)
        print('\nSave: %s' % self.log_dir)
        print(self.var_use)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=conf.param['train']['lr'], weight_decay=conf.param['train']['wd'])
        # return Adam(self.parameters(), lr=conf.param['train']['lr'])

def main():
    lightning_data = LightningData(trainset, valset, testset)
    lightning_model = LightningModel(conf.log_dir, wandb_logger, conf.gpu_id)

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=conf.param['train']['early_stop'],
                                        verbose=False,
                                        strict=False,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath=conf.log_dir, save_top_k=1, monitor='val_loss', mode='min')
    
    trainer = Trainer(
        gpus=[conf.gpu_id],
        max_epochs=conf.param['train']['epoch'],
        precision=conf.param['train']['precision'],
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=wandb_logger,
        default_root_dir=conf.log_dir,
        # track_grad_norm=2 
    )

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)


if __name__ == '__main__':
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description='Air Fusion & Imputation')
    parser.add_argument('--config', type=str, default='config/conf.yaml', help='config file')
    parser.add_argument('--gpu-id', type=int, default=11, help='')
    parser.add_argument('--project-name', type=str, default='air_trash', help='')
    parser.add_argument('--group-name', type=str, default='group_trash', help='')
    parser.add_argument('--run-idx', type=int, default=0, help='')
    parser.add_argument('--model-use', type=str, default="MTCity_GC_GRU", help='')
    parser.add_argument('--preiod-select', type=str, default="p5", help='')
    parser.add_argument('--seed', type=int, default=0, help='')

    parser.add_argument('--enc-len', type=int, default=8, help='')
    parser.add_argument('--dec-len', type=int, default=24, help='')

    parser.add_argument('--epoch', type=int, default=500, help='')
    parser.add_argument('--early-stop', type=int, default=50, help='')
    parser.add_argument('--batch-size', type=int, default=64, help='')
    parser.add_argument('--wd', type=float, default=0.00001, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--hid-dim', type=int, default=16, help='')
    parser.add_argument('--mt-hid-dim', type=int, default=16, help='')
    parser.add_argument('--gnn-out-dim', type=int, default=16, help='')
    parser.add_argument('--c-weight', type=float, default=1.3, help='')
    parser.add_argument('--k-hop', type=int, default=1, help='')
    parser.add_argument('--time-interval', type=int, default=3, help='1 or 3')
    parser.add_argument('--cheb_norm', type=str, default='rw', help='')
    def arg_as_list(s):                                                            
        v = ast.literal_eval(s)                                                    
        if type(v) is not list:                                                    
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
        return v   
    parser.add_argument('--interval-set', type=arg_as_list, default=[1, 2, 4, 8], help='')

    wb_config = parser.parse_args()

    ### wandb_run_name命名
    if (wb_config.model_use == 'MTMSGNN') & (wb_config.k_hop == 1):
        wandb_run_name = 'MTGNN' + '-' + str(wb_config.enc_len) + '-' + str(wb_config.dec_len) + '-' + time.strftime("%Y%m%d%H%M%S", time.localtime())
    else:
        wandb_run_name = wb_config.model_use + '-' + str(wb_config.enc_len) + '-' + str(wb_config.dec_len) + '-' + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if (wb_config.model_use == 'MTMSGNN'):
        wandb_run_name = wandb_run_name + '-k-' + str(wb_config.k_hop) + '-interval-' + '.'.join(str(e) for e in wb_config.interval_set)

    wandb_run_name = wandb_run_name + '_' + str(wb_config.run_idx)

    ### !!!不需要这些 sweep自动init 自动记录hyparam
    # sweep自动运行python xxx.py --xxx x进行传参修改args ---sweep采用
    # wandb.init(config=wb_config, project=wb_config.project_name, group=wb_config.group_name, name=wandb_run_name)
    # wb_config = wandb.config

    ### yaml作为default argparse进一步修改某些值
    conf = Config(wb_config)

    rawdata = RawData(conf)
    if conf.param['city_data_use_flag'] == True:
        trainset = HazeCoarseDataset(rawdata, 'train')
        valset = HazeCoarseDataset(rawdata, 'val')
        testset = HazeCoarseDataset(rawdata, 'test')
    else:
        trainset = HazeDataset(rawdata, 'train')
        valset = HazeDataset(rawdata, 'val')
        testset = HazeDataset(rawdata, 'test')

    # conf.info['group_name'] = 'h' + str(conf.param['train']['hid_dim']) + '-' + str(conf.param['train']['mt_hid_dim']) + '_g' + str(conf.param['train']['gnn_out_dim']) + '_lr' + str(conf.param['train']['lr']) + '_wd' + str(conf.param['train']['wd']) + '_interval-set' + str(conf.param['train']['interval_set']) + '_' + str(conf.param['train']['model_use'])

    ### repeats_run: 遍历seed
    for seed in range(4,5):
        conf.param['train']['seed'] = seed

        pl.utilities.seed.seed_everything(seed) # 直接使用预设的seed
    
        wandb_logger = WandbLogger(name=wandb_run_name+ '_seed' + str(seed), group=conf.info['group_name'], project=conf.info['wandb_proj_name'], save_dir=conf.log_dir, settings=wandb.Settings(start_method="fork")) # 需配合wandb.finish()否则只能启动一次 

        lightning_data = LightningData(trainset, valset, testset)
        lightning_model = LightningModel(conf.log_dir, wandb_logger, conf.param['train'], conf.gpu_id)

        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            patience=conf.param['train']['early_stop'],
                                            verbose=False,
                                            strict=False,
                                            mode='min')

        checkpoint_callback = ModelCheckpoint(dirpath=conf.log_dir, save_top_k=1, monitor='val_loss', mode='min')
        
        trainer = Trainer(
            gpus=[conf.gpu_id],
            max_epochs=conf.param['train']['epoch'],
            precision=conf.param['train']['precision'],
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=wandb_logger,
            default_root_dir=conf.log_dir,
        )


        trainer.fit(lightning_model, lightning_data)
        trainer.test(lightning_model, datamodule=lightning_data)
        wandb.finish()
    
    # ### 正常运行 只取一次seed sweep需要配置config
    # seed = conf.param['train']['seed']
    # pl.utilities.seed.seed_everything(seed) # 选用conf中的seed

    # ### sweep时也要考虑seed, 不同seed合并为1个group_name，到时候比较时选取group取均值, 否则sweep根本就是不稳定下的sweep, 缺乏参考性
    # conf.info['group_name'] = 'h' + str(conf.param['train']['hid_dim']) + '-' + str(conf.param['train']['mt_hid_dim']) + '_g' + str(conf.param['train']['gnn_out_dim']) + '_lr' + str(conf.param['train']['lr']) + '_wd' + str(conf.param['train']['wd']) + '_interval-set' + str(conf.param['train']['interval_set']) + '_' + str(conf.param['train']['model_use'])
        
    # wandb_logger = WandbLogger(name=wandb_run_name+ '_seed' + str(seed), group=conf.info['group_name'], project=conf.info['wandb_proj_name'], save_dir=conf.log_dir, settings=wandb.Settings(start_method="fork")) # 需配合wandb.finish()否则只能启动一次  # log_dir 被 wandb.init()取代?

    # lightning_data = LightningData(trainset, valset, testset)
    # lightning_model = LightningModel(conf.log_dir, wandb_logger, conf.param['train'], conf.gpu_id)

    # early_stop_callback = EarlyStopping(monitor='val_loss',
    #                                     patience=conf.param['train']['early_stop'],
    #                                     verbose=False,
    #                                     strict=False,
    #                                     mode='min')

    # checkpoint_callback = ModelCheckpoint(dirpath=conf.log_dir, save_top_k=1, monitor='val_loss', mode='min')
    
    # trainer = Trainer(
    #     gpus=[conf.gpu_id],
    #     max_epochs=conf.param['train']['epoch'],
    #     precision=conf.param['train']['precision'],
    #     callbacks=[early_stop_callback, checkpoint_callback],
    #     logger=wandb_logger,
    #     default_root_dir=conf.log_dir,
    # )


    # trainer.fit(lightning_model, lightning_data)
    # trainer.test(lightning_model, datamodule=lightning_data)
    # wandb.finish()