# For relative import
import os
import sys
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

import torch
from torch import nn

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        # newgate = torch.tanh(i_n + (resetgate * h_n))
        print("没写完")
        newgate = (input)(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(-1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)


class GRU(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        hid_dim,
        use_met,
        use_time,
        hist_len,
        pred_len,
    ):
        super(GRU, self).__init__()

        self.hid_dim = hid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_met = use_met
        self.use_time = use_time
        self.hist_len = hist_len,
        self.pred_len = pred_len,

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.gru_cell = GRUCell(self.hid_dim, self.hid_dim)

    def forward(self, model_input):

        enc, dec = model_input

        enc_air = enc[0]
        enc_misc = enc[1:]

        if self.use_met or self.use_time:
            dec = torch.cat(dec, dim=-1)
            enc_misc = torch.cat(enc_misc, dim=-1)

        batch_size = enc_air.size(0)
        node_num = enc_air.size(2)

        h0 = torch.zeros(batch_size * node_num, self.hid_dim)
        hn = h0.type_as(enc_air)

        for i in range(self.hist_len[0]-1):
            if enc_misc == []:
                x_i = enc_air[:, i]
            else:
                x_i = torch.cat([enc_misc[:, i], enc_air[:, i]], dim=-1)

            x_i = self.fc_in(x_i)
            hn = self.gru_cell(x_i, hn)

        dec_aqi = []
        dec_aqi_i = enc_air[:, -1]

        for i in range(self.pred_len[0]):

            if dec == []:
                x_i = dec_aqi_i
            else:
                if i == 0:
                    x_i = torch.cat([enc_misc[:, -1], dec_aqi_i], dim=-1)
                else:
                    x_i = torch.cat([dec[:, i-1], dec_aqi_i], dim=-1)

            x_i = self.fc_in(x_i)
            hn = self.gru_cell(x_i, hn)
            xn_i = hn.view(batch_size, node_num, self.hid_dim)
            dec_aqi_i = self.fc_out(xn_i)
            dec_aqi.append(dec_aqi_i)

        dec_aqi = torch.stack(dec_aqi, dim=1)

        return dec_aqi
