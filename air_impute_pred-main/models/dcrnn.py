from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys

from scipy.sparse import linalg
import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict

from torch_geometric.utils import dense_to_sparse, get_laplacian, to_dense_adj



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    '''
    wrapper function for endless data loader.
    '''
    for loader in repeat(data_loader):
        yield from loader


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_random_walk_matrix_pyg(adj_mx):

    edge_idx, edge_attr = dense_to_sparse(adj_mx)
    edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr, 'rw')
    # edge_idx_l, edge_attr_l = get_laplacian(edge_idx, edge_attr)
    A_hat = to_dense_adj(edge_idx_l, edge_attr=edge_attr_l)[0]
    A_hat[A_hat <= 10e-5] = 10e-5

    edge_idx, edge_attr = dense_to_sparse(A_hat)
    return edge_idx, edge_attr


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data



class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class DiffusionGraphConv(BaseModel):
    def __init__(self, input_dim, hid_dim, num_nodes, max_diffusion_step, output_dim, bias_start=0.0, device='cuda', filter_type='dual_random_walk'):
        super(DiffusionGraphConv, self).__init__()

        self.device = device

        # TODO
        # self.num_matrices = len(supports) * max_diffusion_step + 1  # Don't forget to add for x itself.

        self.num_matrices = 2 * max_diffusion_step + 1

        self.filter_type = filter_type
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        # self._supports = supports
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

        self.to(self.device)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, forward_supports, inputs, state, output_size, bias_start=0.0):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs:
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]
        # dtype = inputs.dtype

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        # fusiongraph
        # adj_mat = self.fusiongraph.graph.A_dist.cpu().data


        if self._max_diffusion_step == 0:
            pass
        else:
            for support in forward_supports:
                # x1 = torch.sparse.mm(support, x0)

                x1 = torch.matmul(support, x0)

                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    # x2 = 2 * torch.sparse.mm(support, x1) - x0

                    x2 = 2 * torch.matmul(support, x1) - x0

                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, output_size)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(BaseModel):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self, device, input_dim, num_units, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, use_gc_for_ru=True, filter_type='laplacian'):
        """
        :param num_units: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        self._supports = []
        supports = []
        # supports = calculate_scaled_laplacian(adj_mat, lambda_max=None)  # scipy coo matrix
        # self._supports = self._build_sparse_matrix(supports).to(device)  # to pytorch sparse tensor

        self.dconv_gate = DiffusionGraphConv(input_dim=input_dim,
                                             hid_dim=num_units, num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2, device=device, filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units, device=device, filter_type=filter_type)
        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)

        self.to(device)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def forward(self, forward_supports, inputs, state):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * num_units)
        :return:
        """
        output_size = 2 * self._num_units
        # we start with bias 1.0 to not reset and not update
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(fn(forward_supports, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        c = self.dconv_candidate(forward_supports, inputs, r * state, self._num_units)  # batch_size, self._num_nodes * output_size
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:
            # apply linear projection to state
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))  # (batch*num_nodes, num_units)
            output = torch.reshape(self.project(output), shape=(batch_size, self.output_size))  # (50, 207*1)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        # shape = L.shape
        # i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        # v = torch.FloatTensor(L.data)
        #
        # # i(2, 1396), v(1396), shape （40，40）
        #
        # return torch.sparse.FloatTensor(i, v, torch.Size(shape))

        node_max = L[0].max() + 1
        shape = (node_max, node_max)
        i = L[0]
        v = L[1]
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


class DCRNNEncoder(BaseModel):
    def __init__(self, device, input_dim, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, filter_type):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers

        # encoding_cells = []
        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(device, input_dim=input_dim, num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))
        self.device = device

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(device, input_dim=hid_dim, num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

        self.to(device)

    def forward(self, forward_supports, inputs, initial_hidden_state):

        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 64, 207, 2)
        # inputs to cell is (batch, num_nodes * input_dim)
        # init_hidden_state should be (num_layers, batch_size, num_nodes*num_units) (2, 64, 207*64)
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))  # (12, 64, 207*2)

        current_inputs = inputs
        output_hidden = []  # the output hidden states, shape (num_layers, batch, outdim)
        for i_layer in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](forward_supports, current_inputs[t, ...], hidden_state)  # (50, 207*64)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(self.device)  # seq_len, B, ...
        # output_hidden: the hidden state of each layer at last time step, shape (num_layers, batch, outdim)
        # current_inputs: the hidden state of the top layer (seq_len, B, outdim)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self._num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # init_states shape (num_layers, batch_size, num_nodes*num_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(BaseModel):
    def __init__(self, device, input_dim, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, filter_type, batch, pred_len):
        super(DCGRUDecoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_nodes = num_nodes  # 207
        self._output_dim = output_dim  # should be 1
        self._num_rnn_layers = num_rnn_layers
        self.batch = batch
        self.pred_len = pred_len

        cell = DCGRUCell(device, input_dim=hid_dim, num_units=hid_dim,
                         max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, filter_type=filter_type)
        cell_with_projection = DCGRUCell(device, input_dim=hid_dim, num_units=hid_dim,
                                         max_diffusion_step=max_diffusion_step,
                                         num_nodes=num_nodes, num_proj=output_dim, filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(DCGRUCell(device, input_dim=input_dim, num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers - 1):
            decoding_cells.append(cell)
        decoding_cells.append(cell_with_projection)
        self.decoding_cells = nn.ModuleList(decoding_cells)

        self.device = device

        self.to(device)

    # def forward(self, inputs, initial_hidden_state):

    def forward(self, forward_supports, go, initial_hidden_state):
        """
        :param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
        :param initial_hidden_state: the last hidden state of the encoder. (num_layers, batch, outdim)
        :param teacher_forcing_ratio:
        :return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 50, 207*1)
        """
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 50, 207, 1)
        # inputs to cell is (batch, num_nodes * input_dim)
        seq_length = self.pred_len  # should be 13
        batch_size = self.batch

        go = torch.reshape(go, (self.batch, -1))

        outputs = torch.zeros(seq_length + 1, batch_size, self._num_nodes*self._output_dim).to(self.device)  # (13, 50, 207*1)

        # 32, 38
        current_input = go  # the first input to the rnn is GO Symbol
        for t in range(1, seq_length):
            # hidden_state = initial_hidden_state[i_layer]  # i_layer=0, 1, ...
            next_input_hidden_state = []
            for i_layer in range(0, self._num_rnn_layers):
                # initial_hidden_state[0] [32, 2432]
                #

                # :param
                # inputs: (B, num_nodes * input_dim)
                # :param
                # state: (B, num_nodes * num_units)



                hidden_state = initial_hidden_state[i_layer]
                # current_input: 1, 32, 38
                # hidden_state:
                output, hidden_state = self.decoding_cells[i_layer](forward_supports, current_input, hidden_state)
                current_input = output  # the input of present layer is the output of last layer
                next_input_hidden_state.append(hidden_state)  # store each layer's hidden state
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
            outputs[t] = output  # store the last layer's output to outputs tensor
            # perform scheduled sampling teacher forcing
            # teacher_force = random.random() < teacher_forcing_ratio  # a bool value
            # current_input = (inputs[t] if teacher_force else output)
            current_input = output

        return outputs


class DCRNN(BaseModel):

    def __init__(self, gpu_id, batch_size, in_dim, out_dim, use_met, use_time, enc_len, dec_len, graph, node_num):

        super(DCRNN, self).__init__()

        device = 'cuda:%d' % gpu_id

        enc_input_dim = in_dim
        dec_input_dim = 1

        self.use_met = use_met
        self.use_time = use_time

        # TODO
        # max_diffusion_step = 2
        # max_diffusion_step = 1
        max_diffusion_step = 1

        num_nodes = node_num
        # num_rnn_layers=3

        seq_len=dec_len
        output_dim=out_dim
        # filter_type='dual_random_walk'
        filter_type='dual_random_walk'
        self.filter_type = filter_type
        # filter_type='laplacian'
        # scaler for data normalization
        # self._scaler = scaler
        self._batch_size = batch_size
        self.device = device

        self.graph = graph

        # max_grad_norm parameter is actually defined in data_kwargs
        self._num_nodes = num_nodes  # should be 207
        self._num_rnn_layers = 2  # should be 2
        self._rnn_units = 64  # should be 64
        self._seq_len = seq_len  # should be 12
        # use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))  # should be true
        self._output_dim = output_dim  # should be 1

        # specify a GO symbol as the start of the decoder
        self.GO_Symbol = torch.zeros(1, batch_size, num_nodes * self._output_dim, 1).to(device)

        self.encoder = DCRNNEncoder(device, input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=self._rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=self._num_rnn_layers, filter_type=filter_type).to(device)
        self.decoder = DCGRUDecoder(device, input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=self._rnn_units,
                                    output_dim=self._output_dim,
                                    num_rnn_layers=self._num_rnn_layers, filter_type=filter_type,
                                    batch=self._batch_size, pred_len=self._seq_len
                                    ).to(device)
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

        self.to(device)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        # shape = L.shape
        # i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        # v = torch.FloatTensor(L.data)
        #
        # # i(2, 1396), v(1396), shape （40，40）
        #
        # return torch.sparse.FloatTensor(i, v, torch.Size(shape))

        node_max = L[0].max() + 1
        shape = (node_max, node_max)
        i = L[0]
        v = L[1]
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    # def forward(self, source, target, teacher_forcing_ratio):

    def forward(self, model_input):

        enc, dec = model_input

        enc_air = enc[0]
        enc_misc = enc[1:]

        if self.use_met or self.use_time:
            dec = torch.cat(dec, dim=-1)
            enc_misc = torch.cat(enc_misc, dim=-1)

        source = enc_air

        adj_mat = self.graph

        supports = []
        forward_supports = []
        if self.filter_type == "laplacian":
            supports.append(calculate_random_walk_matrix_pyg(adj_mat, lambda_max=None))
        elif self.filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix_pyg(adj_mat).T)
        elif self.filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix_pyg(adj_mat))
            supports.append(calculate_random_walk_matrix_pyg(adj_mat.T))
            #
            # supports.append(calculate_random_walk_matrix(adj_mat))
            # supports.append(calculate_random_walk_matrix(adj_mat.T))

        else:
            supports.append(calculate_scaled_laplacian(adj_mat))
        for support in supports:
            forward_supports.append(self._build_sparse_matrix(support).to(self.device))  # to PyTorch sparse tensor


        # the size of source/target would be (64, 12, 207, 2)
        source = torch.transpose(source, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(self._batch_size).to(self.device)

        # last hidden state of the encoder is the context
        context, _ = self.encoder(forward_supports, source, init_hidden_state)  # (num_layers, batch, outdim)

        outputs = self.decoder(forward_supports, self.GO_Symbol, context)
        # the elements of the first time step of the outputs are all zeros.
        return outputs[1:, :, :].permute((1, 0, 2))[..., None]  # (seq_length, batch_size, num_nodes*output_dim)  (12, 64, 207*1)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def num_nodes(self):
        return self._num_nodes
