import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from tqdm import tqdm

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):

        super(GCN, self).__init__()

        self.dropout = dropout

        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 256)
        self.gc3 = GraphConvolution(256, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, nclass)

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # prev = 0
        # y = []
        # for idx in idx_map:
        #   y.append(torch.mean(x[prev:idx_map[idx]], 0))
        #   prev = idx_map[idx]
        # y = torch.stack(y, 0)

        # x = x[:num_customers]

        # y = []
        # for X in x:
        #   X = F.relu(self.fc1(X))
        #   X = F.dropout(X, self.dropout, training=self.training)
        #   X = F.softmax(self.fc2(X), dim=0)
        #   y.append(X)
        # y = torch.stack(y, 0)

        y = torch.mean(x, 0)

        y = F.relu(self.fc1(y))
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, self.dropout, training=self.training)
        y = F.softmax(self.fc3(y), dim=0)

        return y
