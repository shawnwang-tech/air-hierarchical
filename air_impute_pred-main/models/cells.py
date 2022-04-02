import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MTGRUCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, interval_set, bias=True): # k: list->代表时间间隔
        super(MTGRUCell, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size *  len(interval_set)
        self.interval_set = interval_set
        self.num_interval = len(interval_set)
        self.hidden_slice_size = int(hidden_size / self.num_interval)
        self.last_hidden_slice_size = hidden_size - (self.hidden_slice_size * (self.num_interval - 1))
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.x2a = nn.Linear(input_size, self.num_interval, bias=bias)
        self.h2a = nn.Linear(hidden_size, self.num_interval, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self): # init params
        i_r, i_i, i_n = self.x2h.weight.chunk(3, 1)
        h_r, h_i, h_n = self.h2h.weight.chunk(3, 1)

        ortho_list = [i_r, i_i, h_r, h_i]
        xavier_list = [i_n, h_n, self.x2a.weight, self.h2a.weight, self.x2a.bias, self.h2a.bias, self.x2h.bias, self.h2h.bias]

        for data in ortho_list:
            nn.init.orthogonal_(data)
        
        for data in xavier_list:
            if data.dim() > 1:
                nn.init.xavier_uniform_(data)
            else:
                nn.init.uniform_(data)
    
    def forward(self, x, hidden, idx):
        x = x.reshape(-1, x.size(-1))

        xa = self.x2a(x) # 用于计算不同时间尺度的权重
        ha = self.h2a(hidden)

        a = torch.softmax(xa + ha, dim=-1)

        def weight_hidden(num_interval, hidden_slice_size, last_hidden_slice_size, hidden, a): # 每个时间尺度按权重更新
            weight_hidden_list = []
            for i in range(num_interval):
                if i == (num_interval-1):
                    weight_hidden_list.append(hidden[:, hidden_slice_size*i : ] * a[:, i][:, None].repeat(1, last_hidden_slice_size))
                else:
                    weight_hidden_list.append(hidden[:, hidden_slice_size*i : hidden_slice_size*(i+1)] * a[:, i][:, None].repeat(1,hidden_slice_size))
            return torch.cat(weight_hidden_list, dim=-1)

        isupdate_set = idx % np.array(self.interval_set) # 判断当前idx下是否更新
        weight_hidden = weight_hidden(self.num_interval, self.hidden_slice_size, self.last_hidden_slice_size, hidden, a)

        gate_x = self.x2h(x)
        gate_h = self.h2h(weight_hidden)

        xz, xr, xh = gate_x.chunk(3, 1)
        hz, hr, hh = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(xr + hr)
        inputgate = torch.sigmoid(xz + hz)
        newgate = torch.tanh(xh + (resetgate * hh))

        update_all_hidden = newgate + inputgate * (weight_hidden - newgate)
        
        def mask_fill(isupdate_set, hidden_slice_size, last_hidden_slice_size, hidden): # 根据isupdate_set保留hidden/更新
            new_hidden_list = []
            # hidden需分为k份考虑, 部分直接copy, 部分update
            for i, isupdate in enumerate(isupdate_set):
                if i == (len(isupdate_set) - 1):
                    if isupdate == 0:
                        new_hidden_list.append(update_all_hidden[:, hidden_slice_size*i : ])
                    else:
                        new_hidden_list.append(hidden[:, hidden_slice_size*i : ])
                else:    
                    if isupdate == 0:
                        new_hidden_list.append(update_all_hidden[:, hidden_slice_size*i : hidden_slice_size*(i+1)])
                    else:
                        new_hidden_list.append(hidden[:, hidden_slice_size*i : hidden_slice_size*(i+1)])
            return torch.cat(new_hidden_list, dim=-1)

        new_hidden = mask_fill(isupdate_set, self.hidden_slice_size, self.last_hidden_slice_size, hidden)

        return new_hidden

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size # 13+13=26
        self.hidden_size = hidden_size # 64
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias) # 26->192;
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias) # 64->192
        self.reset_parameters()

    def reset_parameters(self): # init params
        i_r, i_i, i_n = self.x2h.weight.chunk(3, 1)
        h_r, h_i, h_n = self.h2h.weight.chunk(3, 1)
        x2h_b = self.x2h.bias
        h2h_b = self.h2h.bias

        ortho_list = [i_r, i_i, h_r, h_i]
        xavier_list = [i_n, h_n, x2h_b, h2h_b]

        for data in ortho_list:
            nn.init.orthogonal_(data)
        
        for data in xavier_list:
            if data.dim() > 1:
                nn.init.xavier_uniform_(data)
            else:
                nn.init.uniform_(data)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1)) # Size([32, 184, 26])->Size([5888, 26])

        gate_x = self.x2h(x) # 26->192
        gate_h = self.h2h(hidden) # 64->192

        gate_x = gate_x.squeeze() # seem no need(no dim's number==1)
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1) # torch.chunk(input, chunks, dim=0); dim (int) – dimension along which to split the tensor
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r) # b ?
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate) # Size([5888, 64])

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