import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn import Linear
from torch.nn import ReLU6
from torch.nn import Sequential
import random
from Model.GCN import GraphConvolution, GCN
import numpy as np


import copy

random.seed(200)

# import GCN (later when using python file)

class Generator(Module):
    def __init__(self,
                 PATH,
                 # C: list,
                 maxNumNodes=50,
                 cuda=True,
                 c=0,
                 hyp1=1, 
                 hyp2=2, 
                 start=None,
                 nfeat=128,
                 dropout=0.1,
                 rollout=10):
        """ 
        :param C: Candidate set of nodes (list)
        :param start: Starting node (defaults to randomised node)
        :param c: Which class to maximize prediction scores when generating
        """
        super(Generator, self).__init__()

        self.nfeat = nfeat
        self.dropout = dropout
        self.c = c
        self.rollout = rollout

        self.fc = Linear(nfeat, 256)
        self.gc1 = GraphConvolution(256, 128)
        self.gc2 = GraphConvolution(128, 64)
        self.gc3 = GraphConvolution(64, 32)

        # MLP1
        # 2 FC layers with hidden dimension 16
        self.mlp1 = Sequential(Linear(32, 16),
                               Linear(16, 1))

        # MLP2
        # 2 FC layers with hidden dimension 24
        self.mlp2 = Sequential(Linear(64, 24),
                               Linear(24, 1))

        # Hyperparameters
        self.hyp1 = hyp1
        self.hyp2 = hyp2
        # self.candidate_set = C
        
        # Starting node is the 0th node
        # self.start = 0
        # Maximum number of nodes is 50
        self.maxNumNodes = maxNumNodes
        
        # Load GCN for calculating reward
        self.model = GCN(nfeat=self.nfeat,
                         nclass=2,
                         dropout=0.1)
        
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()
        
        for param in self.model.parameters():
          param.requires_grad = False
        if cuda:
            self.args_cuda = True
            self.cuda()
            self.model.cuda()
        else:
            self.args_cuda = False
                
    def set_graph(self, adj, feat, num_nodes):
        """
        Reset g.G to default graph with only start node
        """
        
        mask_start = torch.BoolTensor([False if i < num_nodes else True for i in range(self.maxNumNodes)])
        
        adj = adj

        feat = feat

        self.G = {'adj': adj, 'feat': feat, 'num_nodes': 1, 'mask_start': mask_start}

    def calculate_loss(self, Rt, p_start, a_start, p_end, a_end, G_t_1):
        """
        Calculated from cross entropy loss (Lce) and reward function (Rt)
        where loss = -Rt*(Lce_start + Lce_end)
        """

        Lce_start = F.cross_entropy(torch.reshape(p_start, (1, self.maxNumNodes)), a_start.unsqueeze(0))
        Lce_end = F.cross_entropy(torch.reshape(p_end, (1, self.maxNumNodes)), a_end.unsqueeze(0))

        return -Rt*(Lce_start + Lce_end)

    def calculate_reward(self, G_t_1):
        """
        Rtr     Calculated from graph rules to encourage generated graphs to be valid
                1. Only one edge to be added between any two nodes
                2. Generated graph cannot contain more nodes than predefined maximum node number
                3. (For chemical) Degree cannot exceed valency
                If generated graph violates graph rule, Rtr = -1

        Rtf     Feedback from trained model
        """

        rtr = self.check_graph_rules(G_t_1)

        rtf, c = self.calculate_reward_feedback(G_t_1)
        rtf_sum = 0
        for m in range(self.rollout):
            p_start, a_start, p_end, a_end, G_t_1 = self.forward(G_t_1)
            rtf_sum += self.calculate_reward_feedback(G_t_1)[0]
        rtf = rtf + rtf_sum * self.hyp1 / self.rollout

        return rtf + self.hyp2 * rtr, self.c == c

    def calculate_reward_feedback(self, G_t_1):
        """
        p(f(G_t_1) = c) - 1/l
        where l denotes number of possible classes for f
        """
        f = self.model(G_t_1['feat'], G_t_1['adj'])
        return f[self.c] - 1/len(f), torch.argmax(f)

    def check_graph_rules(self, G_t_1):
        """
        For USER PREDICTION, there are no known graph rules
        """
        return 0
        
    def forward(self, G_in):
        G = copy.deepcopy(G_in)

        x = G['feat'].detach().clone()
        adj = G['adj'].detach().clone()

        x = F.relu6(self.fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu6(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu6(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu6(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        p_start = self.mlp1(x)
        if self.args_cuda:
            G['mask_start'] = G['mask_start'].cuda()
        p_start = p_start.masked_fill(G['mask_start'].unsqueeze(1), 0)
        p_start = F.softmax(p_start, dim=0)
        a_start_idx = torch.argmax(p_start.masked_fill(G['mask_start'].unsqueeze(1), -1))
        
        # broadcast
        x1, x2 = torch.broadcast_tensors(x, x[a_start_idx])
        x = torch.cat((x1, x2), 1) # cat increases dim from 32 to 64

        mask_end = torch.BoolTensor([False for i in range(self.maxNumNodes)])
        # mask_end[:self.maxNumNodes] = False
        mask_end[a_start_idx] = True
        if self.args_cuda:
            mask_end = mask_end.cuda()
        
        p_end = self.mlp2(x)
        p_end = p_end.masked_fill(mask_end.unsqueeze(1), 0)
        p_end = F.softmax(p_end, dim=0)
        a_end_idx = torch.argmax(p_end.masked_fill(mask_end.unsqueeze(1), -1))

        # Return new G
        # If a_end_idx is not masked, node exists in graph, no new node added
        if G['mask_start'][a_end_idx] == False:
            G['adj'][a_end_idx][a_start_idx] = 1
            G['adj'][a_start_idx][a_end_idx] = 1
            
            # Update degrees
            # G['degrees'][a_start_idx] += 1
            # G['degrees'][G['num_nodes']] += 1
        else:
            # Add node
            G['feat'][G['num_nodes']] = G['feat'][a_end_idx]
            # Add edge
            G['adj'][G['num_nodes']][a_start_idx] = 1
            G['adj'][a_start_idx][G['num_nodes']] = 1
            # Update degrees
            # G['degrees'][a_start_idx] += 1
            # G['degrees'][G['num_nodes']] += 1

            # Update start mask
            G_mask_start_copy = G['mask_start'].detach().clone()
            G_mask_start_copy[G['num_nodes']] = False
            G['mask_start'] = G_mask_start_copy
            
            G['num_nodes'] += 1

        return p_start, a_start_idx, p_end, a_end_idx, G
