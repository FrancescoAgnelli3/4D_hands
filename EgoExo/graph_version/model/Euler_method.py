from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch.nn import Module, Linear, Identity
from torch_geometric.nn import TAGConv
from torch_geometric.data import Data
from typing import Optional
import torch_geometric.nn


class TemporalGraphEuler(Module):
    '''
        h^l_v = h^(l-1)_v + e * activ_fun(-L X W)
    '''
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 delta_t: int,
                 hidden_dim: Optional[int] = None, 
                 K: int = 2,
                 normalization: bool = True, #Optional[str] = 'sym',
                 epsilon: float = 0.1,
                 activ_fun: Optional[str] = 'tanh',
                 use_previous_state: bool = False,
                 bias: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.delta_t = delta_t
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.K = K
        self.normalization = normalization
        self.activ_fun = getattr(torch, activ_fun) if activ_fun is not None else Identity()
        self.bias = bias
        self.use_previous_state = use_previous_state

        self.LayerNorm = nn.LayerNorm(self.hidden_dim)
        self.LayerNorm_out = nn.LayerNorm(self.output_dim)
        self.ReLU = nn.ReLU()


        inp = self.input_dim
        if self.hidden_dim is not None:
            self.emb = nn.Sequential(Linear(self.input_dim, self.hidden_dim))
            inp = self.hidden_dim

        self.conv = TAGConv(in_channels = inp,
                            out_channels = inp,
                            K = self.K,
                            normalize = self.normalization,
                            bias = self.bias)

        self.readout = nn.Sequential(Linear(inp, self.output_dim))
        self.pooling = torch_geometric.nn.global_mean_pool


    def forward(self, data, batch, prev_h: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = data
        edge_index = torch_geometric.nn.knn_graph(data, 10, batch)

        # Build (node, timestamp) encoding
        h = self.emb(x)
        x = self.readout(h)

        if self.use_previous_state and prev_h is not None: 
            h = h + prev_h

        for _ in range(self.delta_t):
            conv = self.conv(h, edge_index)
            h = h + self.epsilon * self.activ_fun(conv)

        y = self.readout(h)
        y = self.pooling(y, batch)

        return y, h